import os, json, pickle, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from dataclasses import dataclass
from typing import List

DIAMOND_ROOT = os.environ.get("DIAMOND_ROOT", "/root/diamond")
sys.path.insert(0, f"{DIAMOND_ROOT}/src")
sys.path.insert(0, "./src")

from models.diffusion.denoiser import Denoiser, DenoiserConfig
from models.diffusion.inner_model import InnerModel, InnerModelConfig
from step3_hook_activations import build_store_and_register

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def build_denoiser(num_actions: int) -> Denoiser:
    cfg = DenoiserConfig(
        sigma_data=0.5,
        sigma_offset_noise=0.3,
        inner_model=InnerModelConfig(
            img_channels=3,
            num_steps_conditioning=4,
            cond_channels=256,
            depths=[2,2,2,2],
            channels=[64,64,64,64],
            attn_depths=[False,False,False,False],
            num_actions=num_actions,
        )
    )
    return Denoiser(cfg)

def load_denoiser(ckpt_path: str) -> Denoiser:
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Strip "denoiser." prefix from all keys
    denoiser_sd = {k.replace("denoiser.", "", 1): v
                   for k, v in state_dict.items() if k.startswith("denoiser.")}
    # Infer num_actions from embedding weight shape
    num_actions = denoiser_sd["inner_model.act_emb.0.weight"].shape[0]
    print(f"    Inferred num_actions={num_actions}")
    denoiser = build_denoiser(num_actions)
    denoiser.load_state_dict(denoiser_sd)
    return denoiser.to(DEVICE).eval()

def load_dataset(game):
    with open(f"data/ram_labels_{game.lower()}.pkl", "rb") as f:
        data = pickle.load(f)
    # img_channels=3 so replicate grayscale to RGB
    frames = torch.from_numpy(data["frames"]).float() / 127.5 - 1.0  # [N,1,84,84]
    frames = frames.expand(-1, 3, -1, -1).contiguous()               # [N,3,84,84]
    labels = torch.from_numpy(data["labels"]).float()
    labels[:, 0] /= 160.0
    labels[:, 1] /= 210.0
    return frames, labels

@torch.no_grad()
def collect_activations(denoiser, frames, batch_size=32):
    store = build_store_and_register(denoiser.inner_model)
    all_acts = {}
    N = frames.shape[0]
    for start in range(0, N, batch_size):
        batch = frames[start:start+batch_size].to(DEVICE)
        B = batch.shape[0]
        # obs: [B, num_steps_conditioning*img_channels, H, W] = [B,12,84,84]
        obs = batch.unsqueeze(1).expand(-1, 4, -1, -1, -1).reshape(B, 12, 84, 84)
        noisy = batch + torch.randn_like(batch)
        sigma = torch.ones(B, device=DEVICE)
        act = torch.zeros(B, 4, dtype=torch.long, device=DEVICE)
        cs = denoiser.compute_conditioners(sigma)
        store.clear()
        _ = denoiser.compute_model_output(noisy, obs, act, cs)
        for name, tensor in store.activations.items():
            all_acts.setdefault(name, []).append(tensor.cpu())
        if start % 3200 == 0:
            print(f"  Activations: {start}/{N}")
    store.remove_hooks()
    return {name: torch.cat(tensors, dim=0) for name, tensors in all_acts.items()}

def train_probe(acts, labels, epochs=50, lr=1e-3):
    N = acts.shape[0]
    dataset = TensorDataset(acts, labels)
    n_test = max(1, int(0.2 * N))
    train_ds, test_ds = random_split(dataset, [N - n_test, n_test])
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    probe = nn.Linear(acts.shape[1], 2).to(DEVICE)
    optim = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        probe.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            nn.MSELoss()(probe(xb), yb).backward()
            optim.step()
    probe.eval()
    xa = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).to(DEVICE)
    ya = torch.stack([test_ds[i][1] for i in range(len(test_ds))]).to(DEVICE)
    with torch.no_grad():
        preds = probe(xa)
        r2 = 1.0 - ((preds-ya)**2).sum().item() / (((ya-ya.mean(0))**2).sum().item()+1e-8)
    return probe.cpu(), r2

def main(game="Pong"):
    os.makedirs("probes", exist_ok=True)
    print(f"\n[1] Loading dataset for {game}...")
    frames, labels = load_dataset(game)
    print(f"    {len(frames)} samples, frame shape: {frames.shape[1:]}")

    print(f"\n[2] Loading checkpoint...")
    ckpt_path = f"/root/diamond/pretrained/atari_100k/models/{game}.pt"
    denoiser = load_denoiser(ckpt_path)
    print(f"    Loaded OK")

    print(f"\n[3] Collecting activations...")
    all_acts = collect_activations(denoiser, frames)
    print(f"    Got {len(all_acts)} layers")

    print(f"\n[4] Training probes...")
    ranking = []
    for layer, acts in all_acts.items():
        probe, r2 = train_probe(acts, labels)
        torch.save({"weight": probe.weight.data, "bias": probe.bias.data,
                    "r2": r2, "layer": layer, "game": game,
                    "act_dim": acts.shape[1]},
                   f"probes/probe_{layer.replace('.','_')}_{game.lower()}.pt")
        ranking.append({"layer": layer, "r2": r2, "act_dim": acts.shape[1]})
        print(f"  {layer:45s}  R²={r2:.4f}")

    ranking.sort(key=lambda x: x["r2"], reverse=True)
    with open(f"probes/layer_ranking_{game.lower()}.json", "w") as f:
        json.dump(ranking, f, indent=2)
    print(f"\nTop 5 layers:")
    for e in ranking[:5]:
        print(f"  {e['layer']:45s}  R²={e['r2']:.4f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="Pong")
    args = p.parse_args()
    main(args.game)
