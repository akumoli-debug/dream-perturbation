"""
Experiment A — Multi-Layer Causal Ablation with Probe-Guided Measurement
==========================================================================
Closes the two biggest remaining gaps:
  1. Tests ALL 8 top-R² layers, not just the best one
  2. Measures causal effect using the TRAINED PROBE on generated frame activations
     instead of noisy row-brightness argmax

This produces a "causal influence profile" — exactly the style of Meng et al. (2022)
ROME paper: a bar chart showing how much causal influence each layer has,
where influence is measured in consistent probe units.

Key design decisions:
  - Measurement: pass generated frame through denoiser, extract activation at same layer,
    run through trained probe → predicted ball_y in [0,1]. This closes the loop.
  - Effect size: report Cohen's d per layer (not just p-value)
  - Negative control: include 2 low-R² layers as controls
  - n=200 pairs per layer (enough for 80% power at expected effect sizes)

Expected results:
  If ANY layer shows Cohen's d > 0.2 → positive causal result, publishable
  If ALL layers show d ≈ 0 → complete layer-wise null, also publishable

HOW TO RUN:
  python3 src/expA_multilayer_causal.py --game Pong --n_pairs 200
  # ~45-60 min
  # Outputs: results/multilayer_causal_profile.json
  #          results/multilayer_causal_profile.png
"""

import os, sys, json, pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split

DIAMOND_ROOT = os.environ.get("DIAMOND_ROOT", "/root/diamond")
sys.path.insert(0, f"{DIAMOND_ROOT}/src")
sys.path.insert(0, "./src")

from models.diffusion.denoiser import Denoiser, DenoiserConfig
from models.diffusion.inner_model import InnerModelConfig
from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
from step3_hook_activations import build_store_and_register

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)

# Top 8 layers by R² from v2 probes + 2 low-R² controls
TARGET_LAYERS = [
    ("u_blocks.0.resblocks.1",  0.618, "top"),    # best layer
    ("d_blocks.3.resblocks.1",  0.598, "top"),
    ("u_blocks.0.resblocks.0",  0.593, "top"),
    ("mid_blocks.resblocks.1",  0.584, "top"),
    ("u_blocks.0.resblocks.2",  0.569, "top"),
    ("d_blocks.3.resblocks.0",  0.567, "top"),
    ("mid_blocks.resblocks.0",  0.555, "top"),
    ("u_blocks.1.resblocks.0",  0.552, "top"),
    ("d_blocks.0.resblocks.0",  0.004, "control"),  # low R² control
    ("u_blocks.3.resblocks.2",  0.040, "control"),  # low R² control
]


def load_denoiser(game="Pong"):
    sd = torch.load(f"/root/diamond/pretrained/atari_100k/models/{game}.pt",
                    map_location="cpu", weights_only=False)
    dsd = {k.replace("denoiser.", "", 1): v
           for k, v in sd.items() if k.startswith("denoiser.")}
    na = dsd["inner_model.act_emb.0.weight"].shape[0]
    cfg = DenoiserConfig(sigma_data=0.5, sigma_offset_noise=0.3,
        inner_model=InnerModelConfig(img_channels=3, num_steps_conditioning=4,
            cond_channels=256, depths=[2,2,2,2], channels=[64,64,64,64],
            attn_depths=[False,False,False,False], num_actions=na))
    d = Denoiser(cfg); d.load_state_dict(dsd)
    return d.to(DEVICE).eval()


def resolve_layer(denoiser, layer_name):
    module = denoiser.inner_model.unet
    for part in layer_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def load_probe(layer_name, game):
    """Load trained linear probe for a given layer."""
    path = f"probes/probe_{layer_name.replace('.','_')}_{game.lower()}.pt"
    if not os.path.exists(path):
        # Try v2 orthogonalized probe
        path = f"probes/probe_{layer_name.replace('.','_')}_{game.lower()}_orth.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No probe found for {layer_name}. Run fix2_train_probes_v2.py first.")
    data = torch.load(path, map_location=DEVICE, weights_only=False)
    W = data["weight"].to(DEVICE)  # [2, C] or [1, C]
    b = data["bias"].to(DEVICE)
    return W, b, data.get("r2", data.get("mean_r2", 0.0))


@torch.no_grad()
def measure_ball_y_with_probe(frame, denoiser, layer_name, probe_W, probe_b):
    """
    PROBE-GUIDED MEASUREMENT: pass generated frame through the denoiser,
    extract the activation at layer_name, apply the trained probe.
    Returns predicted ball_y in [0,1].
    This closes the measurement loop — same representation, same metric.
    """
    store = build_store_and_register(denoiser.inner_model)
    B = frame.shape[0]
    # Build dummy context (repeat frame 4 times — good enough for measurement)
    obs = frame.unsqueeze(1).expand(-1, 4, -1, -1, -1).reshape(B, 12, 84, 84)
    sigma = torch.ones(B, device=DEVICE) * 0.1  # low sigma → near-clean frame
    act = torch.zeros(B, 4, dtype=torch.long, device=DEVICE)
    cs = denoiser.compute_conditioners(sigma)
    store.clear()
    noisy = frame + torch.randn_like(frame) * 0.1
    _ = denoiser.compute_model_output(noisy, obs, act, cs)
    activation = store.activations[layer_name].to(DEVICE)  # [B, C]
    store.remove_hooks()

    # Apply probe: W [2,C] × activation [B,C]^T + b [2]
    if probe_W.shape[0] == 2:
        pred = (activation @ probe_W[1]) + probe_b[1]  # ball_y row
    else:
        pred = (activation @ probe_W[0]) + probe_b[0]
    return pred.cpu()  # [B] predicted ball_y in [0,1]


class PatchSampler(DiffusionSampler):
    """Patches target layer activation from rollout B into rollout A."""
    def __init__(self, denoiser, cfg, target_module):
        super().__init__(denoiser, cfg)
        self.target_module = target_module
        self._act_A = {}
        self._act_B = {}

    def _cap_A(self, m, i, o): self._act_A["v"] = o.detach().clone(); return o
    def _cap_B(self, m, i, o): self._act_B["v"] = o.detach().clone(); return o
    def _patch(self, m, i, o): return self._act_B["v"] if "v" in self._act_B else o

    @torch.no_grad()
    def sample_patched(self, obs_A, act_A, obs_B, act_B):
        device = obs_A.device
        b, t, c, h, w = obs_A.size()
        oAf = obs_A.reshape(b, t*c, h, w)
        oBf = obs_B.reshape(b, t*c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn/(len(self.sigmas)-1), 2**0.5-1)
        xA = torch.randn(b, c, h, w, device=device)
        xB = torch.randn(b, c, h, w, device=device)
        xP = xA.clone()

        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            gamma = gamma_ if self.cfg.s_tmin<=sigma<=self.cfg.s_tmax else 0
            sh = sigma*(gamma+1)
            if gamma > 0:
                n = torch.randn_like(xA)*self.cfg.s_noise*(sh**2-sigma**2)**0.5
                xA = xA + n; xB = xB + torch.randn_like(xB)*self.cfg.s_noise*(sh**2-sigma**2)**0.5
                xP = xP + n.clone()

            hA = self.target_module.register_forward_hook(self._cap_A)
            dA = self.denoiser.denoise(xA, sigma, oAf, act_A); hA.remove()

            hB = self.target_module.register_forward_hook(self._cap_B)
            dB = self.denoiser.denoise(xB, sigma, oBf, act_B); hB.remove()

            hP = self.target_module.register_forward_hook(self._patch)
            dP = self.denoiser.denoise(xP, sigma, oAf, act_A); hP.remove()

            dt = next_sigma - sh
            xA = xA + (xA-dA)/sh*dt
            xB = xB + (xB-dB)/sh*dt
            xP = xP + (xP-dP)/sh*dt

        return xA, xB, xP


def cohens_d(vals_patched, vals_base):
    """Cohen's d = (mean_patched - mean_base) / pooled_std"""
    n1, n2 = len(vals_patched), len(vals_base)
    m1, m2 = np.mean(vals_patched), np.mean(vals_base)
    s1, s2 = np.std(vals_patched, ddof=1), np.std(vals_base, ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (m1 - m2) / (pooled_std + 1e-8)


def bootstrap_ci(shift, n_boot=2000):
    boots = [np.mean(np.random.choice(shift, len(shift), replace=True))
             for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main(game="Pong", n_pairs=200):
    print(f"\n{'='*65}")
    print(f" Multi-Layer Causal Ablation — {game}  (n={n_pairs} per layer)")
    print(f"{'='*65}")

    with open(f"data/context_labels_{game.lower()}.pkl","rb") as f:
        data = pickle.load(f)
    contexts = torch.from_numpy(data["contexts"]).float()/127.5-1.0
    contexts = contexts.unsqueeze(2).expand(-1,-1,3,-1,-1).contiguous()
    ball_y = torch.from_numpy(data["labels"][:,1]).float()

    top_idx = ((ball_y > 10) & (ball_y < 50)).nonzero(as_tuple=True)[0]
    bot_idx = ((ball_y > 150) & (ball_y < 200)).nonzero(as_tuple=True)[0]
    n_pairs = min(n_pairs, len(top_idx), len(bot_idx))
    print(f"  Pairs available: top={len(top_idx)}, bottom={len(bot_idx)}, using={n_pairs}")

    torch.manual_seed(42); np.random.seed(42)
    top_perm = top_idx[torch.randperm(len(top_idx))[:n_pairs]]
    bot_perm = bot_idx[torch.randperm(len(bot_idx))[:n_pairs]]

    denoiser = load_denoiser(game)
    sampler_cfg = DiffusionSamplerConfig(num_steps_denoising=16)
    act = torch.zeros(1, 4, dtype=torch.long, device=DEVICE)

    results = {}
    print(f"\n  {'Layer':40s}  {'R²':>6}  {'Type':>7}  {'CohenD':>8}  {'CI_lo':>8}  {'CI_hi':>8}  {'p≈0':>6}")

    for layer_name, probe_r2, layer_type in TARGET_LAYERS:
        # Load probe for this layer
        try:
            probe_W, probe_b, _ = load_probe(layer_name, game)
        except FileNotFoundError:
            print(f"  {layer_name:40s}  — probe not found, skipping")
            continue

        target_module = resolve_layer(denoiser, layer_name)
        sampler = PatchSampler(denoiser, sampler_cfg, target_module)

        y_A_base, y_B_base, y_patched = [], [], []

        for i in range(n_pairs):
            obs_A = contexts[top_perm[i].item():top_perm[i].item()+1].to(DEVICE)
            obs_B = contexts[bot_perm[i].item():bot_perm[i].item()+1].to(DEVICE)

            gA, gB, gP = sampler.sample_patched(obs_A, act, obs_B, act)

            # PROBE-GUIDED MEASUREMENT
            yA = measure_ball_y_with_probe(gA, denoiser, layer_name, probe_W, probe_b)
            yB = measure_ball_y_with_probe(gB, denoiser, layer_name, probe_W, probe_b)
            yP = measure_ball_y_with_probe(gP, denoiser, layer_name, probe_W, probe_b)

            y_A_base.append(float(yA[0]))
            y_B_base.append(float(yB[0]))
            y_patched.append(float(yP[0]))

        arr_A = np.array(y_A_base)
        arr_B = np.array(y_B_base)
        arr_P = np.array(y_patched)
        shift = arr_P - arr_A

        d = cohens_d(arr_P, arr_A)
        ci_lo, ci_hi = bootstrap_ci(shift)
        p_approx = float((np.array([np.mean(np.random.choice(shift, len(shift), replace=True))
                          for _ in range(1000)]) <= 0).mean())

        results[layer_name] = {
            "r2": probe_r2, "type": layer_type,
            "mean_A": float(arr_A.mean()), "mean_B": float(arr_B.mean()),
            "mean_patched": float(arr_P.mean()),
            "mean_shift": float(shift.mean()), "std_shift": float(shift.std()),
            "cohens_d": d, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_approx": p_approx
        }

        print(f"  {layer_name:40s}  {probe_r2:>6.3f}  {layer_type:>7}  "
              f"{d:>+8.4f}  {ci_lo:>+8.5f}  {ci_hi:>+8.5f}  {p_approx:>6.4f}")

    # ── Plot causal influence profile ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sorted_layers = sorted(results.items(), key=lambda x: x[1]["r2"], reverse=True)
    layer_labels  = [l.replace(".", "\n") for l, _ in sorted_layers]
    cohens_ds     = [v["cohens_d"] for _, v in sorted_layers]
    r2s           = [v["r2"] for _, v in sorted_layers]
    colors        = ["steelblue" if v["type"] == "top" else "lightcoral"
                     for _, v in sorted_layers]

    # Panel 1: Cohen's d per layer
    ax = axes[0]
    bars = ax.bar(range(len(layer_labels)), cohens_ds, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.2, color="green", linestyle="--", alpha=0.5, label="d=0.2 (small effect)")
    ax.axhline(-0.2, color="green", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(layer_labels)))
    ax.set_xticklabels(layer_labels, fontsize=6)
    ax.set_ylabel("Cohen's d (patched vs base)")
    ax.set_title(f"Causal Influence Profile — {game}\n(blue=high-R², red=control layers)")
    ax.legend(fontsize=8)

    # Panel 2: Cohen's d vs probe R²
    ax = axes[1]
    ax.scatter(r2s, cohens_ds, c=["steelblue" if v["type"]=="top" else "lightcoral"
                                    for _, v in sorted_layers], s=80, zorder=3)
    for (l, v), d_val in zip(sorted_layers, cohens_ds):
        ax.annotate(l.split(".")[-2][:8], (v["r2"], d_val), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.2, color="green", linestyle="--", alpha=0.4)
    ax.set_xlabel("Probe R² (spatial encoding quality)")
    ax.set_ylabel("Cohen's d (causal influence)")
    ax.set_title(f"Does probe R² predict causal influence?\n{game}")

    # Add correlation
    r = np.corrcoef(r2s, cohens_ds)[0,1]
    ax.text(0.05, 0.95, f"Pearson r = {r:.3f}", transform=ax.transAxes,
            fontsize=9, verticalalignment='top')

    fig.suptitle("Causal Influence Profile: Which Layers Drive Spatial Generation?",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig("results/multilayer_causal_profile.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved results/multilayer_causal_profile.png")

    # Summary
    top_results = {l: v for l, v in results.items() if v["type"] == "top"}
    ctrl_results = {l: v for l, v in results.items() if v["type"] == "control"}
    max_d = max(top_results.values(), key=lambda x: abs(x["cohens_d"]))
    mean_ctrl_d = np.mean([v["cohens_d"] for v in ctrl_results.values()])

    print(f"\n{'='*65}")
    print(f" SUMMARY")
    print(f"{'='*65}")
    print(f"  Largest Cohen's d (top layers): {max(abs(v['cohens_d']) for v in top_results.values()):.4f}")
    print(f"  Mean Cohen's d (control layers): {mean_ctrl_d:.4f}")
    print(f"  Correlation R² vs Cohen's d: {np.corrcoef(r2s, cohens_ds)[0,1]:.4f}")

    significant = [l for l, v in top_results.items() if v["ci_lo"] > 0 or v["ci_hi"] < 0]
    if significant:
        print(f"\n  CAUSAL SIGNAL FOUND at: {significant}")
        print(f"  → Diffusion world model has causally functional spatial representations")
    else:
        print(f"\n  NULL: No layer shows Cohen's d with CI excluding zero")
        print(f"  → Complete layer-wise confirmation of manifold resistance to patching")

    with open("results/multilayer_causal_profile.json", "w") as f:
        json.dump({"game": game, "n_pairs": n_pairs, "layers": results,
                   "pearson_r_r2_vs_d": float(np.corrcoef(r2s, cohens_ds)[0,1])}, f, indent=2)
    print(f"\n  Saved results/multilayer_causal_profile.json")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="Pong")
    p.add_argument("--n_pairs", type=int, default=200)
    args = p.parse_args()
    main(args.game, args.n_pairs)
