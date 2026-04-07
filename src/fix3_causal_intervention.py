"""
Fix 3 — Causal Interchange Intervention
==========================================
This is the experiment that transforms the project from
"we measured something correlated with position" to
"we found a causally functional spatial representation."

THE EXPERIMENT (causal interchange / activation patching):
  1. Run rollout A: DIAMOND generates next frame from context_A
     (e.g. ball near top, ball_y_A ≈ 30px)
  2. Run rollout B: DIAMOND generates next frame from context_B
     (e.g. ball near bottom, ball_y_B ≈ 150px)
  3. PATCH: during rollout A's denoising, replace the activation at
     u_blocks.0.resblocks.1 with the activation from rollout B
  4. MEASURE: does the generated frame now show the ball near the bottom?

If YES → the representation is causally functional, not just correlated
If NO  → the world model corrects the patch (consistent with steering results)

Either outcome is a clean scientific contribution.

HOW TO RUN:
  # Requires fix1 data and v2 probes
  python3 src/fix3_causal_intervention.py --game Pong --n_pairs 50

  # Takes ~15-20 minutes for 50 pairs
  # Output:
  #   results/causal_intervention_results.json
  #   results/causal_patch_examples/*.png  (visual examples)
"""

import os, sys, json, pickle
import numpy as np
import torch
from PIL import Image

DIAMOND_ROOT = os.environ.get("DIAMOND_ROOT", "/root/diamond")
sys.path.insert(0, f"{DIAMOND_ROOT}/src")
sys.path.insert(0, "./src")

from models.diffusion.denoiser import Denoiser, DenoiserConfig
from models.diffusion.inner_model import InnerModelConfig
from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_LAYER = "u_blocks.0.resblocks.1"


def load_denoiser(game="Pong"):
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
    return d.to(DEVICE).eval()


def resolve_layer(denoiser, layer_name):
    module = denoiser.inner_model.unet
    for part in layer_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


class ActivationPatchingSampler(DiffusionSampler):
    """
    Runs two samplers simultaneously.
    At every denoising step, replaces sampler_A's target layer activation
    with sampler_B's activation.
    Returns three generated frames: base_A, base_B, patched_A.
    """
    def __init__(self, denoiser, cfg, target_module):
        super().__init__(denoiser, cfg)
        self.target_module = target_module
        self._captured_B = {}
        self._patching = False

    def _capture_hook(self, module, input, output):
        """Captures activation from rollout B."""
        self._captured_B["act"] = output.detach().clone()
        return output

    def _patch_hook(self, module, input, output):
        """Replaces activation in rollout A with captured B activation."""
        if "act" in self._captured_B:
            return self._captured_B["act"]
        return output

    @torch.no_grad()
    def sample_with_patch(self, obs_A, act_A, obs_B, act_B):
        """
        Runs A and B in parallel, patching A's activation with B's at each step.
        Returns: (gen_A_base, gen_B_base, gen_A_patched)
        """
        device = obs_A.device
        b, t, c, h, w = obs_A.size()
        obs_A_flat = obs_A.reshape(b, t*c, h, w)
        obs_B_flat = obs_B.reshape(b, t*c, h, w)

        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas)-1), 2**0.5-1)

        # Separate noise initializations
        x_A_base    = torch.randn(b, c, h, w, device=device)
        x_B_base    = torch.randn(b, c, h, w, device=device)
        x_A_patched = x_A_base.clone()  # starts same as A

        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sh = sigma * (gamma + 1)

            if gamma > 0:
                eps = torch.randn_like(x_A_base) * self.cfg.s_noise
                noise_boost = eps * (sh**2 - sigma**2)**0.5
                x_A_base    = x_A_base    + noise_boost
                x_B_base    = x_B_base    + torch.randn_like(x_B_base)*self.cfg.s_noise*(sh**2-sigma**2)**0.5
                x_A_patched = x_A_patched + noise_boost.clone()

            # Step 1: Run B normally, capture its activation
            capture_hook = self.target_module.register_forward_hook(self._capture_hook)
            den_B = self.denoiser.denoise(x_B_base, sigma, obs_B_flat, act_B)
            capture_hook.remove()

            # Step 2: Run A normally (baseline)
            den_A_base = self.denoiser.denoise(x_A_base, sigma, obs_A_flat, act_A)

            # Step 3: Run A with B's activation patched in
            patch_hook = self.target_module.register_forward_hook(self._patch_hook)
            den_A_patched = self.denoiser.denoise(x_A_patched, sigma, obs_A_flat, act_A)
            patch_hook.remove()

            # Euler update for all three
            dt = next_sigma - sh
            x_A_base    = x_A_base    + (x_A_base    - den_A_base)    / sh * dt
            x_B_base    = x_B_base    + (x_B_base    - den_B)         / sh * dt
            x_A_patched = x_A_patched + (x_A_patched - den_A_patched) / sh * dt

        return x_A_base, x_B_base, x_A_patched


def measure_ball_y(frame_tensor):
    """
    Rough ball position: brightest pixel row in playfield [15:75].
    frame_tensor: [3, 84, 84] in [-1,1]
    Returns: y position in pixels (0=top)
    """
    gray = frame_tensor.mean(0)  # [84,84]
    playfield = gray[15:75, :]   # [60, 84]
    # Row with highest mean brightness
    row_brightness = playfield.mean(dim=1)  # [60]
    return int(row_brightness.argmax().item()) + 15


def to_img(t):
    return ((t.cpu().float().clamp(-1,1)+1)/2*255).byte().numpy().transpose(1,2,0)


def main(game="Pong", n_pairs=50):
    os.makedirs("results/causal_patch_examples", exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Causal Interchange Intervention — {game}")
    print(f"{'='*60}")

    # Load data
    print("\n[1] Loading consecutive-context data...")
    ctx_path = f"data/context_labels_{game.lower()}.pkl"
    if not os.path.exists(ctx_path):
        raise FileNotFoundError(f"Run fix1_consecutive_frames.py first")

    with open(ctx_path, "rb") as f:
        data = pickle.load(f)

    contexts_np = data["contexts"]   # [N, 4, 84, 84] uint8
    ball_y_np   = data["labels"][:, 1]  # [N] raw pixels 0-210

    # Convert to tensor
    contexts = torch.from_numpy(contexts_np).float()/127.5-1.0  # [N,4,84,84]
    contexts = contexts.unsqueeze(2).expand(-1,-1,3,-1,-1).contiguous()  # [N,4,3,84,84]
    ball_y   = torch.from_numpy(ball_y_np).float()

    print(f"    {len(contexts)} frames loaded")
    print(f"    Ball_y range: [{ball_y.min():.0f}, {ball_y.max():.0f}] px")

    # Load denoiser
    print("\n[2] Loading denoiser...")
    denoiser = load_denoiser(game)
    target_module = resolve_layer(denoiser, TARGET_LAYER)
    cfg = DiffusionSamplerConfig(num_steps_denoising=16)
    sampler = ActivationPatchingSampler(denoiser, cfg, target_module)

    # Select pairs: A has ball near top (y < 50), B has ball near bottom (y > 150)
    top_idx = (ball_y < 50).nonzero(as_tuple=True)[0]
    bot_idx = (ball_y > 150).nonzero(as_tuple=True)[0]
    print(f"\n[3] Selected pairs:")
    print(f"    'Top' frames (ball_y < 50):    {len(top_idx)} available")
    print(f"    'Bottom' frames (ball_y > 150): {len(bot_idx)} available")
    n_pairs = min(n_pairs, len(top_idx), len(bot_idx))
    print(f"    Running {n_pairs} pairs")

    act = torch.zeros(1, 4, dtype=torch.long, device=DEVICE)

    # Results storage
    ball_y_A_base    = []
    ball_y_B_base    = []
    ball_y_A_patched = []
    ball_y_A_label   = []
    ball_y_B_label   = []

    print("\n[4] Running causal interventions...")
    for i in range(n_pairs):
        idx_A = top_idx[i].item()
        idx_B = bot_idx[i].item()

        obs_A = contexts[idx_A:idx_A+1].to(DEVICE)  # [1,4,3,84,84]
        obs_B = contexts[idx_B:idx_B+1].to(DEVICE)

        gen_A, gen_B, gen_patched = sampler.sample_with_patch(obs_A, act, obs_B, act)

        y_A     = measure_ball_y(gen_A[0])
        y_B     = measure_ball_y(gen_B[0])
        y_patch = measure_ball_y(gen_patched[0])

        ball_y_A_base.append(y_A)
        ball_y_B_base.append(y_B)
        ball_y_A_patched.append(y_patch)
        ball_y_A_label.append(float(ball_y[idx_A]))
        ball_y_B_label.append(float(ball_y[idx_B]))

        if i % 10 == 0:
            print(f"  Pair {i:3d}: A_label={ball_y[idx_A]:.0f}px  "
                  f"B_label={ball_y[idx_B]:.0f}px  | "
                  f"gen_A={y_A}  gen_B={y_B}  gen_patched={y_patch}  "
                  f"(shift={y_patch-y_A:+d}px)")

        # Save visual examples for first 8 pairs
        if i < 8:
            input_A = contexts[idx_A, -1]  # last frame of context A
            input_B = contexts[idx_B, -1]
            row = np.concatenate([
                to_img(input_A), to_img(input_B),
                to_img(gen_A[0]), to_img(gen_B[0]), to_img(gen_patched[0])
            ], axis=1)
            img = Image.fromarray(row).resize(
                (row.shape[1]*4, row.shape[0]*4), Image.NEAREST
            )
            img.save(f"results/causal_patch_examples/pair_{i:02d}.png")

    # Analysis
    arr_A     = np.array(ball_y_A_base)
    arr_B     = np.array(ball_y_B_base)
    arr_patch = np.array(ball_y_A_patched)

    shift = arr_patch - arr_A
    # Bootstrap 95% CI on mean shift
    boot_means = [np.mean(np.random.choice(shift, len(shift), replace=True))
                  for _ in range(1000)]
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    print(f"\n{'='*60}")
    print(f" RESULTS")
    print(f"{'='*60}")
    print(f"  Mean ball_y in gen_A (base):     {arr_A.mean():.1f} ± {arr_A.std():.1f} px")
    print(f"  Mean ball_y in gen_B (base):     {arr_B.mean():.1f} ± {arr_B.std():.1f} px")
    print(f"  Mean ball_y in gen_A (patched):  {arr_patch.mean():.1f} ± {arr_patch.std():.1f} px")
    print(f"\n  Mean shift (patched - base_A):   {shift.mean():+.1f} px")
    print(f"  95% CI (bootstrap):              [{ci_lo:+.1f}, {ci_hi:+.1f}] px")
    print(f"\n  Interpretation:")
    if ci_lo > 0:
        print(f"  CAUSAL: Patching B's activation into A shifts ball DOWN")
        print(f"  → The representation IS causally functional")
    elif ci_hi < 0:
        print(f"  CAUSAL (reverse): Patching B's activation into A shifts ball UP")
        print(f"  → The representation IS causally functional (opposite direction)")
    elif abs(shift.mean()) < 1.0:
        print(f"  NULL: No significant shift — denoising manifold corrects the patch")
        print(f"  → Confirms steering resistance finding with causal evidence")
    else:
        print(f"  PARTIAL: Some shift but CI includes zero — inconclusive")

    # Save results
    results = {
        "n_pairs": n_pairs,
        "mean_shift_px": float(shift.mean()),
        "std_shift_px": float(shift.std()),
        "ci_95_lo": float(ci_lo),
        "ci_95_hi": float(ci_hi),
        "mean_A_base": float(arr_A.mean()),
        "mean_B_base": float(arr_B.mean()),
        "mean_A_patched": float(arr_patch.mean()),
        "all_shifts": shift.tolist(),
    }
    with open("results/causal_intervention_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results/causal_intervention_results.json")
    print(f"  Saved results/causal_patch_examples/pair_00..07.png")
    print(f"  Panels: [input_A | input_B | gen_A | gen_B | gen_A_patched]")
    print(f"  If working: gen_A_patched should look like gen_B's ball position")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="Pong")
    p.add_argument("--n_pairs", type=int, default=50)
    args = p.parse_args()
    main(args.game, args.n_pairs)
