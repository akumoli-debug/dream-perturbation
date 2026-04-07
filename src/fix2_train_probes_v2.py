"""
Fix 2 — Probe Training with Real Consecutive Context + Seeds + CIs
====================================================================
PROBLEMS FIXED vs step4_train_probes.py:
  1. Uses real consecutive 4-frame context (from fix1_consecutive_frames.py)
     instead of repeating a single frame
  2. Runs 5 seeds per layer, reports mean R² ± std
  3. Sets random seeds for reproducibility
  4. Uses act shape [B, 4] matching DIAMOND's actual interface

HOW TO RUN:
  # Run fix1 first to get the context data
  python3 src/fix1_consecutive_frames.py

  # Then run this
  python3 src/fix2_train_probes_v2.py --game Pong
  python3 src/fix2_train_probes_v2.py --game Breakout

  # Takes ~30-45 minutes per game (5 seeds × all layers)
  # Output: probes/layer_ranking_v2_pong.json  (with mean ± std R²)
"""

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
N_SEEDS = 5


def load_dataset(game):
    """
    Load consecutive-context dataset from fix1.
    Returns:
      contexts: [N, 4, 3, 84, 84] float32 in [-1,1]  (RGB, 4 consecutive frames)
      labels:   [N, 2] float32 normalised to [0,1]
    """
    path = f"data/context_labels_{game.lower()}.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run fix1_consecutive_frames.py first."
        )
    with open(path, "rb") as f:
        data = pickle.load(f)

    # contexts: [N, 4, 84, 84] uint8 → float32 [-1,1], replicate to 3 channels
    ctx = torch.from_numpy(data["contexts"]).float() / 127.5 - 1.0  # [N,4,84,84]
    ctx = ctx.unsqueeze(2).expand(-1, -1, 3, -1, -1).contiguous()   # [N,4,3,84,84]

    labels = torch.from_numpy(data["labels"]).float()
    labels[:, 0] /= 160.0
    labels[:, 1] /= 210.0

    return ctx, labels


def load_denoiser(game):
    sd = torch.load(
        f"/root/diamond/pretrained/atari_100k/models/{game}.pt",
        map_location="cpu", weights_only=False
    )
    dsd = {k.replace("denoiser.", "", 1): v
           for k, v in sd.items() if k.startswith("denoiser.")}
    na = dsd["inner_model.act_emb.0.weight"].shape[0]
    cfg = DenoiserConfig(
        sigma_data=0.5, sigma_offset_noise=0.3,
        inner_model=InnerModelConfig(
            img_channels=3, num_steps_conditioning=4,
            cond_channels=256, depths=[2,2,2,2], channels=[64,64,64,64],
            attn_depths=[False,False,False,False], num_actions=na
        )
    )
    d = Denoiser(cfg)
    d.load_state_dict(dsd)
    return d.to(DEVICE).eval(), na


@torch.no_grad()
def collect_activations(denoiser, contexts, batch_size=32):
    """
    Collect activations using REAL consecutive context.
    contexts: [N, 4, 3, 84, 84] — 4 distinct frames per sample
    """
    cache_path = "probes/acts_v2_u_blocks_0_resblocks_1_pong.pt"
    # We collect ALL layers so we only need one pass
    store = build_store_and_register(denoiser.inner_model)
    all_acts = {}
    N = len(contexts)

    for start in range(0, N, batch_size):
        batch_ctx = contexts[start:start+batch_size].to(DEVICE)  # [B,4,3,84,84]
        B = batch_ctx.shape[0]

        # This is the correct obs format: [B, 4*3, 84, 84] = [B, 12, 84, 84]
        # Using REAL consecutive frames, not repeated single frame
        obs = batch_ctx.reshape(B, 12, 84, 84)

        # Use the last frame as noisy input (what DIAMOND would predict next from)
        last_frame = batch_ctx[:, -1]  # [B, 3, 84, 84]
        noisy = last_frame + torch.randn_like(last_frame)

        sigma = torch.ones(B, device=DEVICE)
        # Action: all NOOPs (same as before — action doesn't affect probe training much)
        act = torch.zeros(B, 4, dtype=torch.long, device=DEVICE)

        cs = denoiser.compute_conditioners(sigma)
        store.clear()
        _ = denoiser.compute_model_output(noisy, obs, act, cs)

        for name, tensor in store.activations.items():
            all_acts.setdefault(name, []).append(tensor.cpu())

        if start % 6400 == 0:
            print(f"  Activations: {start}/{N}")

    store.remove_hooks()
    return {name: torch.cat(tensors, dim=0) for name, tensors in all_acts.items()}


def train_probe_seeded(acts, labels, seed, epochs=50, lr=1e-3):
    """Train one probe with a fixed seed. Returns R² on held-out test set."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = len(acts)
    dataset = TensorDataset(acts, labels)
    n_test = max(1, int(0.2 * N))
    train_ds, test_ds = random_split(
        dataset, [N - n_test, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
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
    # Efficient test evaluation using DataLoader
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    preds_list, labels_list = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds_list.append(probe(xb.to(DEVICE)).cpu())
            labels_list.append(yb)
    preds = torch.cat(preds_list)
    ya = torch.cat(labels_list)

    ss_res = ((preds - ya) ** 2).sum().item()
    ss_tot = ((ya - ya.mean(0)) ** 2).sum().item()
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    return r2, probe.cpu()


def main(game="Pong"):
    os.makedirs("probes", exist_ok=True)

    print(f"\n[1] Loading consecutive-context dataset for {game}...")
    contexts, labels = load_dataset(game)
    print(f"    {len(contexts)} samples, context shape: {contexts.shape[1:]}")

    print(f"\n[2] Loading denoiser...")
    denoiser, na = load_denoiser(game)
    print(f"    num_actions={na}")

    print(f"\n[3] Collecting activations with REAL consecutive context...")
    all_acts = collect_activations(denoiser, contexts)
    print(f"    Got {len(all_acts)} layers")

    print(f"\n[4] Training probes with {N_SEEDS} seeds per layer...")
    print(f"    {'Layer':45s}  {'mean R²':>8}  {'std R²':>8}  {'min R²':>8}  {'max R²':>8}")

    ranking = []
    for layer, acts in all_acts.items():
        r2_scores = []
        for seed in range(N_SEEDS):
            r2, _ = train_probe_seeded(acts, labels, seed=seed)
            r2_scores.append(r2)

        mean_r2 = float(np.mean(r2_scores))
        std_r2  = float(np.std(r2_scores))
        min_r2  = float(np.min(r2_scores))
        max_r2  = float(np.max(r2_scores))

        ranking.append({
            "layer":   layer,
            "mean_r2": mean_r2,
            "std_r2":  std_r2,
            "min_r2":  min_r2,
            "max_r2":  max_r2,
            "all_r2":  r2_scores,
            "act_dim": acts.shape[1],
        })
        print(f"    {layer:45s}  {mean_r2:>8.4f}  {std_r2:>8.4f}  "
              f"{min_r2:>8.4f}  {max_r2:>8.4f}")

    ranking.sort(key=lambda x: x["mean_r2"], reverse=True)

    out_path = f"probes/layer_ranking_v2_{game.lower()}.json"
    with open(out_path, "w") as f:
        json.dump(ranking, f, indent=2)

    print(f"\n=== Top 5 layers for {game} (with 95% CI) ===")
    for e in ranking[:5]:
        ci = 1.96 * e["std_r2"] / (N_SEEDS ** 0.5)
        print(f"  {e['layer']:45s}  R²={e['mean_r2']:.4f} ± {ci:.4f} (95% CI)")

    print(f"\nFull ranking saved → {out_path}")

    # Compare with original single-seed results
    orig_path = f"probes/layer_ranking_{game.lower()}.json"
    if os.path.exists(orig_path):
        with open(orig_path) as f:
            orig = {e["layer"]: e["r2"] for e in json.load(f)}
        print(f"\n=== Comparison: fabricated obs vs real consecutive context ===")
        print(f"  {'Layer':45s}  {'Old R²':>8}  {'New R²':>8}  {'Δ':>8}")
        for e in ranking[:10]:
            old = orig.get(e["layer"], float("nan"))
            delta = e["mean_r2"] - old
            print(f"  {e['layer']:45s}  {old:>8.4f}  {e['mean_r2']:>8.4f}  {delta:>+8.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="Pong")
    args = p.parse_args()
    main(args.game)
