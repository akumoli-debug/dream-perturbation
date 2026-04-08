"""
Experiment B — Cross-Game Dose-Response + Causal Efficiency Ratio
==================================================================
Tests whether probe R² predicts causal influence across games.

Hypothesis: causal efficiency = (dose-response effect at α=1.0) / (probe R²)
            is approximately constant across Pong and Breakout.

If confirmed: probe R² is a cheap screening metric for world model steerability.
If disconfirmed: spatial encoding quality and causal controllability are decoupled.

Both outcomes are publishable because they directly answer:
"Can you predict world model steerability without running the full causal experiment?"

Builds on existing dose-response infrastructure (add1_dose_response.py).
Runs Pong + Breakout with identical methodology and reports:
  - Dose-response effect size per game
  - Probe R² per game (from existing v2 results)
  - Causal efficiency ratio = effect / R²
  - Whether the ratio is consistent across games

HOW TO RUN:
  # Pong probes already exist. Run Breakout probes first if not done:
  python3 src/fix2_train_probes_v2.py --game Breakout

  # Then run this:
  python3 src/expB_cross_game_efficiency.py --n_pairs 500
  # ~50 minutes (250 pairs × 2 games × 5 alphas)
  # Outputs:
  #   results/cross_game_dose_response.json
  #   results/cross_game_dose_response.png
  #   results/causal_efficiency.json
"""

import os, sys, json, pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

DIAMOND_ROOT = os.environ.get("DIAMOND_ROOT", "/root/diamond")
sys.path.insert(0, f"{DIAMOND_ROOT}/src")
sys.path.insert(0, "./src")

from models.diffusion.denoiser import Denoiser, DenoiserConfig
from models.diffusion.inner_model import InnerModelConfig
from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
os.makedirs("results", exist_ok=True)


def load_denoiser(game):
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
    m = denoiser.inner_model.unet
    for p in layer_name.split("."):
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m


def get_best_layer_r2(game):
    """Get the best layer and its R² from v2 probe results."""
    path = f"probes/layer_ranking_v2_{game.lower()}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run fix2_train_probes_v2.py --game {game} first")
    with open(path) as f:
        ranking = json.load(f)
    best = ranking[0]
    return best["layer"], best["mean_r2"], best["std_r2"]


class MixedSampler(DiffusionSampler):
    """Mixes activations: alpha*act_B + (1-alpha)*act_A at target layer."""
    def __init__(self, denoiser, cfg, target_module, alpha=0.0):
        super().__init__(denoiser, cfg)
        self.target_module = target_module
        self.alpha = alpha
        self._aA, self._aB = {}, {}

    def _cA(self, m, i, o): self._aA["v"] = o.detach().clone(); return o
    def _cB(self, m, i, o): self._aB["v"] = o.detach().clone(); return o
    def _mix(self, m, i, o):
        if "v" in self._aA and "v" in self._aB:
            return (1-self.alpha)*self._aA["v"] + self.alpha*self._aB["v"]
        return o

    @torch.no_grad()
    def sample_mixed(self, obs_A, act_A, obs_B, act_B):
        device = obs_A.device
        b, t, c, h, w = obs_A.size()
        oAf = obs_A.reshape(b, t*c, h, w)
        oBf = obs_B.reshape(b, t*c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn/(len(self.sigmas)-1), 2**0.5-1)
        xA = torch.randn(b, c, h, w, device=device)
        xB = torch.randn(b, c, h, w, device=device)
        xM = xA.clone()
        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            gamma = gamma_ if self.cfg.s_tmin<=sigma<=self.cfg.s_tmax else 0
            sh = sigma*(gamma+1)
            if gamma > 0:
                n = torch.randn_like(xA)*self.cfg.s_noise*(sh**2-sigma**2)**0.5
                xA=xA+n; xB=xB+torch.randn_like(xB)*self.cfg.s_noise*(sh**2-sigma**2)**0.5; xM=xM+n.clone()
            hA = self.target_module.register_forward_hook(self._cA)
            dA = self.denoiser.denoise(xA, sigma, oAf, act_A); hA.remove()
            hB = self.target_module.register_forward_hook(self._cB)
            dB = self.denoiser.denoise(xB, sigma, oBf, act_B); hB.remove()
            hM = self.target_module.register_forward_hook(self._mix)
            dM = self.denoiser.denoise(xM, sigma, oAf, act_A); hM.remove()
            dt = next_sigma - sh
            xA=xA+(xA-dA)/sh*dt; xB=xB+(xB-dB)/sh*dt; xM=xM+(xM-dM)/sh*dt
        return xA, xB, xM


def measure_bottom_bias(frame):
    """Brightness(bottom half playfield) - Brightness(top half)."""
    gray = frame.mean(0)
    return float(gray[47:79,:].mean() - gray[15:47,:].mean())


def bootstrap_ci(vals, n=2000):
    boots = [np.mean(np.random.choice(vals, len(vals), replace=True)) for _ in range(n)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_dose_response(game, n_pairs, best_layer):
    """Run dose-response experiment for one game. Returns dict of results by alpha."""
    print(f"\n  Running dose-response for {game} at layer {best_layer}...")

    with open(f"data/context_labels_{game.lower()}.pkl","rb") as f:
        data = pickle.load(f)
    contexts = torch.from_numpy(data["contexts"]).float()/127.5-1.0
    contexts = contexts.unsqueeze(2).expand(-1,-1,3,-1,-1).contiguous()
    ball_y = torch.from_numpy(data["labels"][:,1]).float()

    # Use game-specific top/bottom splits based on actual ball_y distribution
    if game == "Breakout":
        # Breakout ball ranges 76-209px, median ~152
        top_idx = ((ball_y > 76) & (ball_y < 115)).nonzero(as_tuple=True)[0]
        bot_idx = ((ball_y >= 175) & (ball_y < 210)).nonzero(as_tuple=True)[0]
    else:
        # Pong ball ranges 0-207px
        top_idx = ((ball_y > 10) & (ball_y < 50)).nonzero(as_tuple=True)[0]
        bot_idx = ((ball_y > 150) & (ball_y < 200)).nonzero(as_tuple=True)[0]
    n_pairs = min(n_pairs, len(top_idx), len(bot_idx))
    print(f"    Top frames: {len(top_idx)}, Bottom: {len(bot_idx)}, Using: {n_pairs}")

    denoiser = load_denoiser(game)
    target = resolve_layer(denoiser, best_layer)
    sampler_cfg = DiffusionSamplerConfig(num_steps_denoising=16)
    act = torch.zeros(1, 4, dtype=torch.long, device=DEVICE)

    torch.manual_seed(42); np.random.seed(42)
    top_p = top_idx[torch.randperm(len(top_idx))[:n_pairs]]
    bot_p = bot_idx[torch.randperm(len(bot_idx))[:n_pairs]]

    by_alpha = {a: [] for a in ALPHAS}
    for i in range(n_pairs):
        obs_A = contexts[top_p[i].item():top_p[i].item()+1].to(DEVICE)
        obs_B = contexts[bot_p[i].item():bot_p[i].item()+1].to(DEVICE)
        for alpha in ALPHAS:
            s = MixedSampler(denoiser, sampler_cfg, target, alpha)
            _, _, gM = s.sample_mixed(obs_A, act, obs_B, act)
            by_alpha[alpha].append(measure_bottom_bias(gM[0]))
        if i % 100 == 0:
            print(f"    Pair {i}/{n_pairs}  α=0.0:{np.mean(by_alpha[0.0]):+.5f}  α=1.0:{np.mean(by_alpha[1.0]):+.5f}")

    stats = {}
    for alpha in ALPHAS:
        vals = np.array(by_alpha[alpha])
        lo, hi = bootstrap_ci(vals)
        stats[alpha] = {"mean": float(vals.mean()), "ci_lo": lo, "ci_hi": hi, "n": n_pairs}

    # Effect size = mean(α=1.0) - mean(α=0.0)
    effect = stats[1.0]["mean"] - stats[0.0]["mean"]
    base_vals = np.array(by_alpha[0.0])
    one_vals  = np.array(by_alpha[1.0])
    pooled_std = np.sqrt((base_vals.std(ddof=1)**2 + one_vals.std(ddof=1)**2) / 2)
    cohens_d = effect / (pooled_std + 1e-8)

    # Bootstrap CI on effect
    boot_effects = [np.mean(np.random.choice(one_vals, len(one_vals), replace=True)) -
                    np.mean(np.random.choice(base_vals, len(base_vals), replace=True))
                    for _ in range(2000)]
    eff_lo, eff_hi = np.percentile(boot_effects, [2.5, 97.5])
    p_val = float((np.array(boot_effects) <= 0).mean())

    print(f"    Effect (α=0→1): {effect:+.6f}  Cohen's d: {cohens_d:+.4f}  p: {p_val:.4f}")

    return {
        "game": game, "best_layer": best_layer, "n_pairs": n_pairs,
        "stats_by_alpha": {str(k): v for k, v in stats.items()},
        "effect": float(effect), "cohens_d": float(cohens_d),
        "effect_ci_lo": float(eff_lo), "effect_ci_hi": float(eff_hi),
        "p_value": p_val,
        "by_alpha_means": {str(a): float(np.mean(by_alpha[a])) for a in ALPHAS},
    }


def main(n_pairs=250):
    print(f"\n{'='*65}")
    print(f" Cross-Game Dose-Response + Causal Efficiency")
    print(f"{'='*65}")

    game_results = {}
    efficiency_data = {}

    for game in ["Pong", "Breakout"]:
        try:
            best_layer, r2_mean, r2_std = get_best_layer_r2(game)
            print(f"\n[{game}] Best layer: {best_layer}  R²={r2_mean:.4f} ± {r2_std:.4f}")
        except FileNotFoundError as e:
            print(f"\n[{game}] {e}")
            continue

        res = run_dose_response(game, n_pairs, best_layer)
        game_results[game] = res

        # Causal efficiency = effect_size / probe_R²
        efficiency = res["effect"] / (r2_mean + 1e-8)
        efficiency_data[game] = {
            "probe_r2": r2_mean, "probe_r2_std": r2_std,
            "effect": res["effect"], "cohens_d": res["cohens_d"],
            "p_value": res["p_value"],
            "causal_efficiency": float(efficiency),
            "best_layer": best_layer,
        }
        print(f"\n  [{game}] Causal efficiency = {efficiency:.6f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"Pong": "steelblue", "Breakout": "coral"}

    # Panel 1: Dose-response curves side by side
    ax = axes[0]
    for game, res in game_results.items():
        means = [res["by_alpha_means"][str(a)] for a in ALPHAS]
        # Normalize to α=0 baseline
        baseline = means[0]
        normalized = [m - baseline for m in means]
        ax.plot(ALPHAS, normalized, "o-", color=colors[game], label=game, linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mixing weight α")
    ax.set_ylabel("Brightness bias shift (α=0 normalized to 0)")
    ax.set_title("Dose-Response Curves by Game")
    ax.legend()

    # Panel 2: Effect size comparison
    ax = axes[1]
    games = list(efficiency_data.keys())
    effects = [efficiency_data[g]["effect"] for g in games]
    r2s = [efficiency_data[g]["probe_r2"] for g in games]
    ax.bar(games, effects, color=[colors[g] for g in games])
    ax.set_ylabel("Dose-response effect (α=0 → α=1)")
    ax.set_title("Effect Size by Game")
    for i, (g, e) in enumerate(zip(games, effects)):
        ax.text(i, e + max(effects)*0.02, f"d={efficiency_data[g]['cohens_d']:+.3f}",
                ha="center", fontsize=9)

    # Panel 3: R² vs effect (the key efficiency question)
    ax = axes[2]
    for game in games:
        ed = efficiency_data[game]
        ax.scatter(ed["probe_r2"], ed["effect"], color=colors[game],
                   s=120, zorder=3, label=game)
        ax.annotate(f"{game}\n(eff={ed['causal_efficiency']:.5f})",
                    (ed["probe_r2"], ed["effect"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)
    ax.set_xlabel("Probe R²")
    ax.set_ylabel("Dose-response effect")
    ax.set_title("Is Probe R² a Proxy for Causal Influence?")
    ax.legend()

    # Add consistency annotation
    if len(efficiency_data) == 2:
        effs = [v["causal_efficiency"] for v in efficiency_data.values()]
        ratio = max(effs) / (min(effs) + 1e-8)
        label = f"Efficiency ratio: {ratio:.2f}x\n{'≈ consistent' if ratio < 2 else '≠ inconsistent'}"
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Cross-Game Causal Efficiency Analysis", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig("results/cross_game_dose_response.png", dpi=150)
    plt.close(fig)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f" CAUSAL EFFICIENCY SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Game':12s}  {'Probe R²':>10}  {'Effect':>12}  {'Cohens_d':>10}  {'Efficiency':>12}")
    for game, ed in efficiency_data.items():
        print(f"  {game:12s}  {ed['probe_r2']:>10.4f}  {ed['effect']:>+12.6f}  "
              f"{ed['cohens_d']:>+10.4f}  {ed['causal_efficiency']:>+12.6f}")

    if len(efficiency_data) == 2:
        effs = list(efficiency_data.values())
        ratio = max(e["causal_efficiency"] for e in effs) / \
                (min(e["causal_efficiency"] for e in effs) + 1e-8)
        print(f"\n  Efficiency ratio (Pong/Breakout): {ratio:.2f}x")
        if ratio < 2.0:
            print("  → CONSISTENT: probe R² is a reliable proxy for causal influence")
            print("    Implication: can screen world model layers without full causal experiment")
        else:
            print("  → INCONSISTENT: probe R² does not reliably predict causal influence")
            print("    Implication: encoding quality and causal controllability are decoupled")

    # Save
    all_results = {"game_results": game_results, "efficiency": efficiency_data}
    with open("results/cross_game_dose_response.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open("results/causal_efficiency.json", "w") as f:
        json.dump(efficiency_data, f, indent=2)
    print(f"\n  Saved results/cross_game_dose_response.json")
    print(f"  Saved results/causal_efficiency.json")
    print(f"  Saved results/cross_game_dose_response.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_pairs", type=int, default=250)
    args = p.parse_args()
    main(args.n_pairs)
