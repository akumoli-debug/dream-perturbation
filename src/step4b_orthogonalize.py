import os, sys, json, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

DIAMOND_ROOT = os.environ.get("DIAMOND_ROOT", "/root/diamond")
sys.path.insert(0, f"{DIAMOND_ROOT}/src")
sys.path.insert(0, "./src")

from models.diffusion.denoiser import Denoiser, DenoiserConfig
from models.diffusion.inner_model import InnerModelConfig
from step3_hook_activations import build_store_and_register

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_LAYER = "u_blocks.0.resblocks.1"
GAME = "Pong"

def load_labels():
    with open("data/ram_labels_pong.pkl","rb") as f:
        data = pickle.load(f)
    frames = data["frames"]
    score_brightness = frames[:,0,:14,:].mean(axis=(1,2)).astype(np.float32) / 255.0
    ball_x = data["labels"][:,0] / 160.0
    ball_y = data["labels"][:,1] / 210.0
    print(f"Corr(score, ball_y): {np.corrcoef(score_brightness, ball_y)[0,1]:.3f}")
    return score_brightness, ball_x, ball_y, data["frames"]

def load_or_collect_activations(denoiser, frames_np, batch_size=32):
    cache = f"probes/acts_{TARGET_LAYER.replace('.','_')}_pong.pt"
    if os.path.exists(cache):
        print(f"Loading cached activations from {cache}")
        return torch.load(cache, weights_only=False)
    print("Collecting activations (no cache)...")
    store = build_store_and_register(denoiser.inner_model)
    frames_t = torch.from_numpy(frames_np).float()/127.5-1.0
    frames_t = frames_t.expand(-1,3,-1,-1).contiguous()
    all_acts = []
    N = len(frames_t)
    for start in range(0, N, batch_size):
        batch = frames_t[start:start+batch_size].to(DEVICE)
        B = batch.shape[0]
        obs = batch.unsqueeze(1).expand(-1,4,-1,-1,-1).reshape(B,12,84,84)
        noisy = batch + torch.randn_like(batch)
        sigma = torch.ones(B, device=DEVICE)
        act = torch.zeros(B,4,dtype=torch.long,device=DEVICE)
        cs = denoiser.compute_conditioners(sigma)
        store.clear()
        with torch.no_grad():
            _ = denoiser.compute_model_output(noisy, obs, act, cs)
        all_acts.append(store.activations[TARGET_LAYER].cpu())
        if start % 16000 == 0:
            print(f"  {start}/{N}")
    store.remove_hooks()
    acts = torch.cat(all_acts, dim=0)
    torch.save(acts, cache)
    print(f"Saved cache → {cache}")
    return acts

def train_probe(acts, labels_np, label_name=""):
    labels = torch.from_numpy(labels_np).float().unsqueeze(1)
    dataset = TensorDataset(acts, labels)
    N = len(dataset)
    n_test = max(1, int(0.2*N))
    train_ds, test_ds = random_split(dataset, [N-n_test, n_test])
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    probe = nn.Linear(acts.shape[1], 1).to(DEVICE)
    optim = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(50):
        probe.train()
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE),yb.to(DEVICE)
            optim.zero_grad()
            nn.MSELoss()(probe(xb),yb).backward()
            optim.step()
    probe.eval()
    xa = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).to(DEVICE)
    ya = torch.stack([test_ds[i][1] for i in range(len(test_ds))]).to(DEVICE)
    with torch.no_grad():
        preds = probe(xa)
        r2 = 1.0 - ((preds-ya)**2).sum().item()/(((ya-ya.mean())**2).sum().item()+1e-8)
    print(f"  [{label_name}] R²={r2:.4f}")
    return probe.cpu().weight.data.squeeze(0)

def gram_schmidt(target, confounds):
    v = target.clone().float()
    for c in confounds:
        c_hat = c.float()/(c.norm()+1e-8)
        v = v - (v @ c_hat)*c_hat
    return v/(v.norm()+1e-8)

def main():
    os.makedirs("probes", exist_ok=True)
    print("[1] Loading labels...")
    score_brightness, ball_x, ball_y, frames_np = load_labels()

    print("\n[2] Loading denoiser...")
    sd = torch.load(f"/root/diamond/pretrained/atari_100k/models/{GAME}.pt",
                    map_location="cpu", weights_only=False)
    denoiser_sd = {k.replace("denoiser.","",1):v for k,v in sd.items() if k.startswith("denoiser.")}
    num_actions = denoiser_sd["inner_model.act_emb.0.weight"].shape[0]
    cfg = DenoiserConfig(sigma_data=0.5, sigma_offset_noise=0.3,
        inner_model=InnerModelConfig(img_channels=3, num_steps_conditioning=4,
            cond_channels=256, depths=[2,2,2,2], channels=[64,64,64,64],
            attn_depths=[False,False,False,False], num_actions=num_actions))
    denoiser = Denoiser(cfg)
    denoiser.load_state_dict(denoiser_sd)
    denoiser = denoiser.to(DEVICE).eval()

    acts = load_or_collect_activations(denoiser, frames_np)
    print(f"Activations: {acts.shape}")

    print("\n[3] Training confound probes...")
    w_score = train_probe(acts, score_brightness, "score_brightness")
    w_ballx = train_probe(acts, ball_x, "ball_x")

    ball_probe = torch.load(
        f"probes/probe_{TARGET_LAYER.replace('.','_')}_{GAME.lower()}.pt",
        weights_only=False)
    w_bally = ball_probe["weight"][1].float()

    print("\n[4] Orthogonalizing...")
    w_orth = gram_schmidt(w_bally, [w_score, w_ballx])

    acts_f = acts.float()
    orig_hat = w_bally/(w_bally.norm()+1e-8)
    proj_orig = (acts_f @ orig_hat).numpy()
    proj_orth = (acts_f @ w_orth).numpy()

    print(f"\n  Cosine(original, orth): {float(orig_hat @ w_orth):.4f}")
    print(f"  Corr(original → score): {np.corrcoef(proj_orig, score_brightness)[0,1]:+.4f}")
    print(f"  Corr(orth    → score):  {np.corrcoef(proj_orth, score_brightness)[0,1]:+.4f}  ← target: ≈0")
    print(f"  Corr(original → ball_y):{np.corrcoef(proj_orig, ball_y)[0,1]:+.4f}")
    print(f"  Corr(orth    → ball_y): {np.corrcoef(proj_orth, ball_y)[0,1]:+.4f}  ← should stay positive")

    out = f"probes/probe_{TARGET_LAYER.replace('.','_')}_{GAME.lower()}_orth.pt"
    torch.save({"weight": w_orth.unsqueeze(0), "bias": torch.zeros(1),
                "r2": ball_probe["r2"], "layer": TARGET_LAYER,
                "game": GAME, "act_dim": acts.shape[1],
                "orthogonalized_against": ["score_brightness","ball_x"]}, out)
    print(f"\nSaved → {out}")

if __name__ == "__main__":
    main()
