import json, os, sys
from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn

DIAMOND_ROOT = os.environ.get("DIAMOND_ROOT", "/root/diamond")
sys.path.insert(0, f"{DIAMOND_ROOT}/src")
sys.path.insert(0, "./src")

@dataclass
class GeometricSteerConfig:
    game: str
    layer_name: str
    steer_axis: int   # 0=x, 1=y
    steer_alpha: float = 2.0
    steer_direction: int = 1
    probes_dir: str = "probes"

class SteeringHook:
    def __init__(self, target_module, vector, alpha, direction):
        self.vector = vector
        self.alpha = alpha
        self.direction = direction
        self._hook = target_module.register_forward_hook(self._fn)

    def _fn(self, module, input, output):
        steer = self.direction * self.alpha * self.vector.view(1, -1, 1, 1)
        return output + steer

    def remove(self):
        self._hook.remove()

def load_steering_vector(cfg, device):
    layer_safe = cfg.layer_name.replace(".", "_")
    path = f"{cfg.probes_dir}/probe_{layer_safe}_{cfg.game.lower()}.pt"
    data = torch.load(path, map_location=device, weights_only=False)
    W = data["weight"].to(device)
    v = W[cfg.steer_axis]
    v = v / (v.norm() + 1e-8)
    print(f"[Steering] Layer: {cfg.layer_name}  R²={data['r2']:.4f}  dim={v.shape[0]}")
    return v

def resolve_layer(denoiser, layer_name):
    module = denoiser.inner_model.unet
    for part in layer_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module

def best_layer_from_ranking(game, probes_dir="probes", min_r2=0.3):
    with open(f"{probes_dir}/layer_ranking_{game.lower()}.json") as f:
        ranking = json.load(f)
    best = next((r for r in ranking if r["r2"] >= min_r2), None)
    if best is None:
        raise ValueError(f"No layer with R²>={min_r2} for {game}")
    return best["layer"], best["r2"]

# ── Quick test without full WorldModelEnv ─────────────────────────────────────

def test_steering_effect(denoiser, cfg, n_frames=10):
    """
    Generates n_frames with and without steering from random noise,
    prints mean absolute pixel difference to confirm steering has an effect.
    """
    device = next(denoiser.parameters()).device
    vector = load_steering_vector(cfg, device)
    target = resolve_layer(denoiser, cfg.layer_name)

    from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
    sampler_cfg = DiffusionSamplerConfig(num_steps_denoising=16)
    sampler = DiffusionSampler(denoiser, sampler_cfg)

    B = n_frames
    # Dummy obs: [B, 12, 84, 84] (4 steps * 3 channels)
    obs = torch.randn(B, 4, 3, 84, 84, device=device)
    act = torch.zeros(B, 4, dtype=torch.long, device=device)

    # Baseline
    with torch.no_grad():
        base, _ = sampler.sample(obs, act)

    print("\nSteering effect (mean |Δframe|):")
    for alpha in [0.5, 1.0, 2.0, 4.0, 8.0]:
        hook = SteeringHook(target, vector, alpha, cfg.steer_direction)
        with torch.no_grad():
            steered, _ = sampler.sample(obs, act)
        hook.remove()
        diff = (steered - base).abs().mean().item()
        print(f"  alpha={alpha:.1f}  |Δ|={diff:.5f}")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, f"{DIAMOND_ROOT}/src")
    from models.diffusion.denoiser import Denoiser, DenoiserConfig
    from models.diffusion.inner_model import InnerModelConfig

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load denoiser
    state_dict = torch.load(
        "/root/diamond/pretrained/atari_100k/models/Pong.pt",
        map_location="cpu", weights_only=False
    )
    denoiser_sd = {k.replace("denoiser.", "", 1): v
                   for k, v in state_dict.items() if k.startswith("denoiser.")}
    num_actions = denoiser_sd["inner_model.act_emb.0.weight"].shape[0]

    from models.diffusion.denoiser import DenoiserConfig
    from models.diffusion.inner_model import InnerModelConfig
    cfg_model = DenoiserConfig(
        sigma_data=0.5, sigma_offset_noise=0.3,
        inner_model=InnerModelConfig(
            img_channels=3, num_steps_conditioning=4, cond_channels=256,
            depths=[2,2,2,2], channels=[64,64,64,64],
            attn_depths=[False,False,False,False], num_actions=num_actions,
        )
    )
    denoiser = Denoiser(cfg_model)
    denoiser.load_state_dict(denoiser_sd)
    denoiser = denoiser.to(DEVICE).eval()

    layer_name, r2 = best_layer_from_ranking("Pong")
    print(f"Best layer: {layer_name}  R²={r2:.4f}")

    steer_cfg = GeometricSteerConfig(
        game="Pong",
        layer_name=layer_name,
        steer_axis=1,       # steer ball_y
        steer_alpha=2.0,
        steer_direction=1,
    )
    test_steering_effect(denoiser, steer_cfg)
