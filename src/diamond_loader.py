"""Load DIAMOND's pretrained Pong denoiser from HuggingFace."""
import sys
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

DIAMOND_ROOT = Path("/workspace/diamond")
if str(DIAMOND_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(DIAMOND_ROOT / "src"))

try:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
except ValueError:
    pass


def _download_game(game: str = "Pong"):
    ckpt = hf_hub_download(repo_id="eloialonso/diamond", filename=f"atari_100k/models/{game}.pt")
    agent_cfg = hf_hub_download(repo_id="eloialonso/diamond", filename="atari_100k/config/agent/default.yaml")
    env_cfg = hf_hub_download(repo_id="eloialonso/diamond", filename="atari_100k/config/env/atari.yaml")
    return Path(ckpt), Path(agent_cfg), Path(env_cfg)


def load_denoiser(game: str = "Pong", device: str = "cuda", num_actions: int = 6, untrained: bool = False):
    """Return (Denoiser module with weights loaded, full agent config)."""
    from models.diffusion.denoiser import Denoiser

    ckpt_path, agent_cfg_path, _env_cfg_path = _download_game(game)

    cfg_dir = agent_cfg_path.parent
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        agent_cfg = compose(config_name=agent_cfg_path.stem)

    OmegaConf.set_struct(agent_cfg, False)
    agent_cfg.denoiser.inner_model.num_actions = num_actions

    # instantiate() returns DenoiserConfig (that's the _target_); build Denoiser from it.
    denoiser_cfg = instantiate(agent_cfg.denoiser)
    denoiser = Denoiser(denoiser_cfg)

    if untrained:
        print("UNTRAINED baseline: skipping weight load, using random init")
        return denoiser.to(device).eval(), agent_cfg
    # Checkpoint is the full Agent state_dict. Extract denoiser.* keys.
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    filt = {k.replace("denoiser.", "", 1): v for k, v in sd.items() if k.startswith("denoiser.")}
    missing, unexpected = denoiser.load_state_dict(filt, strict=False)
    if missing:
        print(f"w: missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"w: unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")

    denoiser = denoiser.to(device).eval()
    return denoiser, agent_cfg


if __name__ == "__main__":
    d, cfg = load_denoiser()
    n_params = sum(p.numel() for p in d.parameters())
    print("Loaded Pong denoiser OK")
    im = cfg.denoiser.inner_model
    print(f"  channels: {im.channels}")
    print(f"  depths: {im.depths}")
    print(f"  num_steps_conditioning: {im.num_steps_conditioning}")
    print(f"  num_actions: {im.num_actions}")
    print(f"  total params: {n_params/1e6:.1f}M")
