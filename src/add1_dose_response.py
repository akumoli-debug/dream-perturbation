"""
Addition 1 — Causal Dose-Response Curve
==========================================
Extends the binary patch (A vs B activation) to a continuous mixing sweep.
At each mixing weight α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}, the denoiser sees:
  mixed_activation = α * act_B + (1-α) * act_A

If the representation is causally functional, the generated frame's spatial
brightness distribution should shift monotonically with α.

This is a much stronger causal claim than a binary patch because:
  1. It characterizes the dose-response relationship
  2. Monotonicity is a necessary condition for clean causal influence
  3. Each α level can be tested independently with its own CI

Expected result:
  α=0.0: gen frame matches rollout A's ball position
  α=1.0: gen frame partially shifts toward rollout B's ball position
  Monotonic intermediate steps: confirms causal gradient, not noise

HOW TO RUN:
  python3 src/add1_dose_response.py --game Pong --n_pairs 500
  # ~25 minutes
  # Outputs:
  #   results/dose_response_results.json
  #   results/dose_response_curve.png
"""

import os, sys, json, pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

DIAMOND_ROOT = os.environ.get("DIAMOND_ROOT", "/root/diamond")
sys.path.insert(0, f"{DIAMOND_ROOT}/src")
sys.path.insert(0, "./src")

from models.diffusion.denoiser import Denoiser, DenoiserConfig
from models.diffusion.inner_model import InnerModelConfig
from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_LAYER = "u_blocks.0.resblocks.1"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


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


class MixedPatchSampler(DiffusionSampler):
    """
    Runs rollout A and B simultaneously.
    At each denoising step, patches A's target activation with
    α * act_B + (1-α) * act_A for a given mixing weight α.
    """
    def __init__(self, denoiser, cfg, target_module, alpha=0.0):
        super().__init__(denoiser, cfg)
        self.target_module = target_module
        self.alpha = alpha
        self._act_A = {}
        self._act_B = {}

    def _capture_A(self, module, input, output):
        self._act_A["v"] = output.detach().clone()
        return output

    def _capture_B(self, module, input, output):
        self._act_B["v"] = output.detach().clone()
        return output

    def _mixed_patch(self, module, input, output):
        if "v" in self._act_A and "v" in self._act_B:
            return (1 - self.alpha) * self._act_A["v"] + self.alpha * self._act_B["v"]
        return output

    @torch.no_grad()
    def sample_mixed(self, obs_A, act_A, obs_B, act_B):
        device = obs_A.device
        b, t, c, h, w = obs_A.size()
        obs_A_flat = obs_A.reshape(b, t*c, h, w)
        obs_B_flat = obs_B.reshape(b, t*c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn/(len(self.sigmas)-1), 2**0.5-1)

        x_A = torch.randn(b, c, h, w, device=device)
        x_B = torch.randn(b, c, h, w, device=device)
        x_mixed = x_A.clone()

        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sh = sigma * (gamma + 1)
            if gamma > 0:
                n = torch.randn_like(x_A) * self.cfg.s_noise * (sh**2 - sigma**2)**0.5
                x_A = x_A + n
                x_B = x_B + torch.randn_like(x_B)*self.cfg.s_noise*(sh**2-sigma**2)**0.5
                x_mixed = x_mixed + n.clone()

            # Capture A's activation
            h_A = self.target_module.register_forward_hook(self._capture_A)
            den_A = self.denoiser.denoise(x_A, sigma, obs_A_flat, act_A)
            h_A.remove()

            # Capture B's activation
            h_B = self.target_module.register_forward_hook(self._capture_B)
            den_B = self.denoiser.denoise(x_B, sigma, obs_B_flat, act_B)
            h_B.remove()

            # Mixed patch on A
            h_mix = self.target_module.register_forward_hook(self._mixed_patch)
            den_mixed = self.denoiser.denoise(x_mixed, sigma, obs_A_flat, act_A)
            h_mix.remove()

            dt = next_sigma - sh
            x_A     = x_A     + (x_A     - den_A)     / sh * dt
            x_B     = x_B     + (x_B     - den_B)     / sh * dt
            x_mixed = x_mixed + (x_mixed - den_mixed)  / sh * dt

        return x_A, x_B, x_mixed


def measure_bottom_bias(frame):
    """Brightness(bottom half) - Brightness(top half) of playfield."""
    gray = frame.mean(0)
    return float(gray[47:79, :].mean() - gray[15:47, :].mean())


def bootstrap_ci(values, n_boot=2000, ci=95):
    boots = [np.mean(np.random.choice(values, len(values), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(boots, (100-ci)/2)
    hi = np.percentile(boots, 100-(100-ci)/2)
    return lo, hi


def main(game="Pong", n_pairs=500):
    os.makedirs("results", exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Causal Dose-Response Curve — {game}  (n={n_pairs} pairs per alpha)")
    print(f"{'='*60}")

    # Load data
    with open(f"data/context_labels_{game.lower()}.pkl","rb") as f:
        data = pickle.load(f)
    contexts = torch.from_numpy(data["contexts"]).float()/127.5-1.0
    contexts = contexts.unsqueeze(2).expand(-1,-1,3,-1,-1).contiguous()
    ball_y = torch.from_numpy(data["labels"][:,1]).float()

    # Strict on-screen filter
    top_idx = ((ball_y > 10) & (ball_y < 50)).nonzero(as_tuple=True)[0]
    bot_idx = ((ball_y > 150) & (ball_y < 200)).nonzero(as_tuple=True)[0]
    n_pairs = min(n_pairs, len(top_idx), len(bot_idx))
    print(f"  Top frames: {len(top_idx)}, Bottom frames: {len(bot_idx)}, Using: {n_pairs}")

    # Load denoiser
    denoiser = load_denoiser(game)
    target_module = resolve_layer(denoiser, TARGET_LAYER)
    sampler_cfg = DiffusionSamplerConfig(num_steps_denoising=16)
    act = torch.zeros(1, 4, dtype=torch.long, device=DEVICE)

    # Shuffle pairs with fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    top_perm = top_idx[torch.randperm(len(top_idx))[:n_pairs]]
    bot_perm = bot_idx[torch.randperm(len(bot_idx))[:n_pairs]]

    # Run dose-response sweep
    results_by_alpha = {alpha: [] for alpha in ALPHAS}

    print(f"\n  Running {n_pairs} pairs × {len(ALPHAS)} alpha values...")
    for i in range(n_pairs):
        idx_A = top_perm[i].item()
        idx_B = bot_perm[i].item()
        obs_A = contexts[idx_A:idx_A+1].to(DEVICE)
        obs_B = contexts[idx_B:idx_B+1].to(DEVICE)

        for alpha in ALPHAS:
            sampler = MixedPatchSampler(denoiser, sampler_cfg, target_module, alpha=alpha)
            _, _, gen_mixed = sampler.sample_mixed(obs_A, act, obs_B, act)
            results_by_alpha[alpha].append(measure_bottom_bias(gen_mixed[0]))

        if i % 100 == 0:
            means = {a: np.mean(results_by_alpha[a]) for a in ALPHAS}
            print(f"  Pair {i:4d}/{n_pairs}  " +
                  "  ".join(f"α={a:.2f}:{means[a]:+.5f}" for a in ALPHAS))

    # Compute statistics
    print(f"\n{'='*60}")
    print(f"  {'α':>6}  {'mean':>10}  {'CI_lo':>10}  {'CI_hi':>10}  {'p≤0':>8}")
    stats = {}
    means_list, ci_lo_list, ci_hi_list = [], [], []
    for alpha in ALPHAS:
        vals = np.array(results_by_alpha[alpha])
        mean = vals.mean()
        lo, hi = bootstrap_ci(vals)
        p_zero = float((np.array([np.mean(np.random.choice(vals, len(vals), replace=True))
                                   for _ in range(2000)]) <= results_by_alpha[0.0][0]).mean())
        # Simpler: fraction of bootstrap means <= baseline mean
        base_vals = np.array(results_by_alpha[0.0])
        boot_diffs = [np.mean(np.random.choice(vals, len(vals), replace=True)) -
                      np.mean(np.random.choice(base_vals, len(base_vals), replace=True))
                      for _ in range(2000)]
        p_vs_base = float((np.array(boot_diffs) <= 0).mean())

        stats[alpha] = {"mean": float(mean), "ci_lo": float(lo), "ci_hi": float(hi),
                        "p_vs_baseline": p_vs_base, "n": len(vals)}
        means_list.append(mean); ci_lo_list.append(lo); ci_hi_list.append(hi)
        print(f"  {alpha:>6.2f}  {mean:>+10.6f}  {lo:>+10.6f}  {hi:>+10.6f}  {p_vs_base:>8.4f}")

    # Check monotonicity
    is_monotonic = all(means_list[i] <= means_list[i+1] for i in range(len(means_list)-1))
    print(f"\n  Monotonically increasing: {is_monotonic}")
    if is_monotonic:
        print("  STRONG CAUSAL: dose-response is monotonic — clean causal gradient")
    else:
        print("  NON-MONOTONIC: mixed causal signal")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ci_err_lo = [m - lo for m, lo in zip(means_list, ci_lo_list)]
    ci_err_hi = [hi - m for m, hi in zip(means_list, ci_hi_list)]
    ax.errorbar(ALPHAS, means_list, yerr=[ci_err_lo, ci_err_hi],
                fmt='o-', capsize=5, color='steelblue', linewidth=2, markersize=6)
    ax.axhline(means_list[0], linestyle='--', color='gray', alpha=0.5, label='α=0 baseline')
    ax.set_xlabel("Mixing weight α (0=pure A, 1=pure B activation)")
    ax.set_ylabel("Bottom-half brightness bias\n(positive = ball lower in frame)")
    ax.set_title(f"Causal Dose-Response: Activation Mixing at {TARGET_LAYER}\n{game}, n={n_pairs} pairs")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/dose_response_curve.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved results/dose_response_curve.png")

    # Save JSON
    with open("results/dose_response_results.json", "w") as f:
        json.dump({"game": game, "n_pairs": n_pairs, "target_layer": TARGET_LAYER,
                   "alphas": ALPHAS, "stats": {str(k): v for k, v in stats.items()},
                   "monotonic": is_monotonic}, f, indent=2)
    print(f"  Saved results/dose_response_results.json")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="Pong")
    p.add_argument("--n_pairs", type=int, default=500)
    args = p.parse_args()
    main(args.game, args.n_pairs)
