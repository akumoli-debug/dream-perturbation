"""
dream_perturbation_experiment_v3.py
====================================
Improved "Dream Perturbation" experiment for DIAMOND world models — v3.

Changes from v2:
  - Saves actor_critic checkpoints after training each agent
  - Adds transfer evaluation (sim-to-real robustness) under 3 novel conditions:
      A. Sticky Actions (repeat_action_probability=0.25)
      B. Observation Noise (Gaussian noise std=0.05 on normalised obs)
      C. Frame-Skip 6 (instead of standard 4)
  - Generates transfer_robustness.png figure
  - Prints transfer summary table

This script:
  1. Loads a pretrained DIAMOND agent checkpoint from HuggingFace
  2. Creates 5 PerturbedWorldModelEnv dream variants
  3. Trains 4 actor-critic agents:
       (a) Baseline            — trains only in the "normal" (unperturbed) dream
       (b) Single-perturb      — trains only in the "mirrored" dream variant
       (c) Multi-dream         — cycles through ALL 5 dream variants each epoch
       (d) Adaptive-multi-dream — adaptive curriculum: spends more time on
                                   variants it's currently BAD at
  4. Saves actor_critic checkpoints after each agent is trained
  5. Evaluates all 4 agents in the REAL Atari environment
  6. Evaluates all 4 agents in HELD-OUT dream variants
  7. Runs transfer tests (3 novel conditions) for all 4 agents
  8. Saves JSON results + generates comparison figures

Run from inside the DIAMOND repository root:
    python path/to/dream_perturbation_experiment_v3.py

Or via the provided shell script:
    bash scripts/run_dream_perturbation.sh
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — DIAMOND's src/ must be on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent  # learning-from-failure/
# We expect the script is run from the DIAMOND repo root OR the repo path
# is passed as --diamond-root. See argument parsing below.

import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Matplotlib (non-interactive backend for headless servers)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dream Perturbation experiment v3 for DIAMOND world models"
    )
    p.add_argument(
        "--diamond-root",
        type=str,
        default=os.environ.get("DIAMOND_ROOT", str(Path.cwd())),
        help="Path to the DIAMOND repository root (default: $DIAMOND_ROOT or CWD)",
    )
    p.add_argument(
        "--game",
        type=str,
        default="Breakout",
        help="Atari game name to use (default: Breakout). Used for checkpoint download and env creation.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to agent checkpoint .pt file. "
             "If not given, downloads the specified game from HuggingFace.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(_REPO_ROOT / "results"),
        help="Directory to write results and figures (default: results/)",
    )
    p.add_argument(
        "--train-steps-per-epoch",
        type=int,
        default=200,
        help="Actor-critic gradient steps per training epoch (default: 200).",
    )
    p.add_argument(
        "--num-epochs",
        type=int,
        default=200,
        help="Number of training epochs per agent (default: 200).",
    )
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Real-env evaluation episodes per agent (default: 50).",
    )
    p.add_argument(
        "--wm-horizon",
        type=int,
        default=50,
        help="Dream horizon (steps per episode in world model, default: 50).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Parallel envs / batch size for actor-critic (default: 16).",
    )
    p.add_argument(
        "--init-collect-steps",
        type=int,
        default=1000,
        help="Steps to collect in real env for world-model initialisation (default: 1000).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed (default: 42).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device string, e.g. 'cuda:0'. Auto-detected if not set.",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Quick smoke-test mode: 10 epochs, 50 steps/epoch, 5 eval episodes.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# DIAMOND import helper
# ---------------------------------------------------------------------------

def setup_diamond_imports(diamond_root: str) -> None:
    """Add DIAMOND's src/ directory to sys.path."""
    src_path = str(Path(diamond_root) / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    print(f"[Setup] DIAMOND src path: {src_path}")


# ---------------------------------------------------------------------------
# Apply required patches to DIAMOND / PyTorch
# ---------------------------------------------------------------------------

def apply_patches() -> None:
    """
    Patch known incompatibilities for newer PyTorch (2.11+):

    1. torch.load: add weights_only=False to suppress FutureWarning and allow
       loading legacy checkpoints that contain custom classes.

    2. BatchSampler.__init__: PyTorch 2.11 changed the Sampler base class
       signature; super().__init__() no longer accepts a `dataset` argument.
       We patch DIAMOND's BatchSampler to call super().__init__() with no args.

    3. OmegaConf resolver: register the "eval" resolver used in trainer.yaml.
    """
    import functools

    # --- Patch 1: torch.load ---
    _original_torch_load = torch.load

    @functools.wraps(_original_torch_load)
    def _patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
        kwargs.setdefault("weights_only", False)
        if pickle_module is not None:
            return _original_torch_load(f, map_location=map_location,
                                        pickle_module=pickle_module, **kwargs)
        return _original_torch_load(f, map_location=map_location, **kwargs)

    torch.load = _patched_torch_load
    print("[Patch] torch.load patched (weights_only=False by default)")

    # --- Patch 2: BatchSampler (applied after DIAMOND is imported) ---
    # Deferred — see _patch_batch_sampler() called after imports.

    # --- Patch 3: OmegaConf resolver ---
    try:
        from omegaconf import OmegaConf
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        print("[Patch] OmegaConf 'eval' resolver registered")
    except Exception as e:
        print(f"[Patch] OmegaConf resolver warning: {e}")


def _patch_batch_sampler() -> None:
    """
    Patch DIAMOND's BatchSampler to be compatible with PyTorch >= 2.11 where
    torch.utils.data.Sampler.__init__() no longer accepts positional args.
    Must be called after 'from data import BatchSampler' has succeeded.
    """
    try:
        from data.batch_sampler import BatchSampler

        _orig_init = BatchSampler.__init__

        def _new_init(self, dataset, rank, world_size, batch_size, seq_length,
                      sample_weights=None, can_sample_beyond_end=False):
            # Call grandparent (torch.utils.data.Sampler) without args
            torch.utils.data.Sampler.__init__(self)
            self.dataset = dataset
            self.rank = rank
            self.world_size = world_size
            self.sample_weights = sample_weights
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.can_sample_beyond_end = can_sample_beyond_end

        BatchSampler.__init__ = _new_init
        print("[Patch] BatchSampler.__init__ patched for PyTorch 2.11+")
    except Exception as e:
        print(f"[Patch] BatchSampler patch skipped: {e}")


# ---------------------------------------------------------------------------
# Checkpoint download
# ---------------------------------------------------------------------------

def download_pretrained_checkpoint(game: str = "Breakout") -> Path:
    """Download the pretrained DIAMOND agent from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download
    print(f"[Download] Fetching pretrained {game} checkpoint from HuggingFace…")
    path = hf_hub_download(
        repo_id="eloialonso/diamond",
        filename=f"atari_100k/models/{game}.pt",
    )
    print(f"[Download] Checkpoint cached at: {path}")
    return Path(path)


def download_pretrained_configs() -> Tuple[Path, Path]:
    """Download agent config + env config from HuggingFace."""
    from huggingface_hub import hf_hub_download
    agent_cfg_path = hf_hub_download(
        repo_id="eloialonso/diamond",
        filename="atari_100k/config/agent/default.yaml",
    )
    env_cfg_path = hf_hub_download(
        repo_id="eloialonso/diamond",
        filename="atari_100k/config/env/atari.yaml",
    )
    return Path(agent_cfg_path), Path(env_cfg_path)


# ---------------------------------------------------------------------------
# Build dataset for world-model burn-in
# ---------------------------------------------------------------------------

def collect_initial_dataset(
    real_env,
    actor_critic,
    dataset,
    num_steps: int,
    device: torch.device,
) -> None:
    """
    Roll out the (possibly random) actor-critic in the real environment to
    build an initial dataset for the world-model burn-in generator.
    """
    from coroutines.collector import make_collector, NumToCollect

    print(f"[Collect] Gathering {num_steps} steps in real environment…")
    collector = make_collector(real_env, actor_critic, dataset, epsilon=1.0)
    collector.send(NumToCollect(steps=num_steps))
    print(f"[Collect] Dataset: {dataset.num_episodes} eps, {dataset.num_steps} steps")


# ---------------------------------------------------------------------------
# World-model environment builder
# ---------------------------------------------------------------------------

def build_world_model_env(
    agent,
    dataset,
    batch_size: int,
    horizon: int,
    num_steps_conditioning: int,
    device: torch.device,
    num_batches_to_preload: int = 3,
    num_workers: int = 0,
):
    """
    Build a WorldModelEnv (unperturbed) from a loaded DIAMOND agent.

    Returns
    -------
    wm_env : WorldModelEnv
    """
    from functools import partial
    from torch.utils.data import DataLoader

    from data import BatchSampler, collate_segments_to_batch
    from envs import WorldModelEnv, WorldModelEnvConfig
    from models.diffusion import DiffusionSamplerConfig
    from hydra.utils import instantiate

    use_cuda = (device.type == "cuda")
    make_data_loader = partial(
        DataLoader,
        dataset=dataset,
        collate_fn=collate_segments_to_batch,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=use_cuda,
        pin_memory_device=str(device) if use_cuda else "",
    )

    seq_length = num_steps_conditioning  # obs context length for actor-critic
    bs = BatchSampler(dataset, rank=0, world_size=1,
                      batch_size=batch_size, seq_length=seq_length)
    dl = make_data_loader(batch_sampler=bs)

    diffusion_sampler_cfg = DiffusionSamplerConfig(num_steps_denoising=3)
    wm_env_cfg = WorldModelEnvConfig(
        horizon=horizon,
        num_batches_to_preload=num_batches_to_preload,
        diffusion_sampler=diffusion_sampler_cfg,
    )

    wm_env = WorldModelEnv(
        denoiser=agent.denoiser,
        rew_end_model=agent.rew_end_model,
        data_loader=dl,
        cfg=wm_env_cfg,
        return_denoising_trajectory=False,
    )
    return wm_env


# ---------------------------------------------------------------------------
# Actor-critic PPO training loop (follows DIAMOND's trainer.py pattern)
# ---------------------------------------------------------------------------

def train_actor_critic_one_epoch(
    actor_critic,
    rl_env,
    optimizer: torch.optim.AdamW,
    grad_acc_steps: int = 1,
    backup_every: int = 15,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    weight_value_loss: float = 1.0,
    weight_entropy_loss: float = 0.01,
    max_grad_norm: Optional[float] = 10.0,
) -> Dict[str, float]:
    """
    Run one epoch of actor-critic training using DIAMOND's env_loop.

    This mirrors Trainer.train_component('actor_critic', steps) but is
    self-contained so we can swap rl_env freely.

    The actor_critic.env_loop must already have been set up by calling
    actor_critic.setup_training(rl_env, loss_cfg).

    Returns a dict of scalar metrics.
    """
    import math
    from torch.distributions.categorical import Categorical
    from models.actor_critic import compute_lambda_returns

    actor_critic.train()
    optimizer.zero_grad()

    metrics_accum = {
        "loss_total": 0.0,
        "loss_actions": 0.0,
        "loss_values": 0.0,
        "loss_entropy": 0.0,
        "policy_entropy": 0.0,
    }

    steps_done = 0
    num_updates = max(1, grad_acc_steps)

    for step_i in range(num_updates):
        # Fetch a trajectory from the env_loop coroutine
        outputs = actor_critic.env_loop.send(backup_every)
        _, act, rew, end, trunc, logits_act, val, val_bootstrap, _ = outputs

        # Compute lambda returns
        lambda_returns = compute_lambda_returns(
            rew, end, trunc, val_bootstrap, gamma, lambda_
        )

        d = Categorical(logits=logits_act)
        entropy = d.entropy().mean()
        advantage = (lambda_returns - val).detach()

        loss_actions = (-d.log_prob(act) * advantage).mean()
        loss_values = weight_value_loss * torch.nn.functional.mse_loss(val, lambda_returns)
        loss_entropy = -weight_entropy_loss * entropy
        loss = loss_actions + loss_values + loss_entropy

        loss.backward()
        steps_done += 1

        metrics_accum["loss_total"] += loss.detach().item()
        metrics_accum["loss_actions"] += loss_actions.detach().item()
        metrics_accum["loss_values"] += loss_values.detach().item()
        metrics_accum["loss_entropy"] += loss_entropy.detach().item()
        metrics_accum["policy_entropy"] += entropy.detach().item() / math.log(2)

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    return {k: v / max(steps_done, 1) for k, v in metrics_accum.items()}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_in_real_env(
    actor_critic,
    real_env,
    num_episodes: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Roll out the actor-critic in a REAL Atari environment and return
    episode-level statistics.

    Returns
    -------
    dict with keys: mean_return, std_return, mean_length, std_length
    """
    from torch.distributions.categorical import Categorical

    actor_critic.eval()
    returns = []
    lengths = []

    lstm_dim = actor_critic.lstm_dim

    # Use batch size 1 for cleaner episode counting
    hx = torch.zeros(1, lstm_dim, device=device)
    cx = torch.zeros(1, lstm_dim, device=device)

    # Reset the real env (gym-style; returns obs, info)
    obs, _ = real_env.reset()

    ep_return = 0.0
    ep_length = 0
    episodes_done = 0

    max_steps = num_episodes * 5000  # safety cap
    steps = 0

    while episodes_done < num_episodes and steps < max_steps:
        with torch.no_grad():
            # obs from Atari env: [1, C, H, W] on device already (TorchEnv)
            logits_act, val, (hx, cx) = actor_critic.predict_act_value(obs, (hx, cx))
        act = Categorical(logits=logits_act).sample()

        obs, rew, end, trunc, info = real_env.step(act)
        ep_return += rew.sum().item()
        ep_length += 1
        steps += 1

        done = torch.logical_or(end, trunc)
        if done.any():
            returns.append(ep_return)
            lengths.append(ep_length)
            episodes_done += 1
            ep_return = 0.0
            ep_length = 0
            hx = hx * 0.0
            cx = cx * 0.0

    if not returns:
        returns = [0.0]
        lengths = [0]

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "num_episodes": len(returns),
    }


def evaluate_in_dream_variant(
    actor_critic,
    perturbed_env,
    num_episodes: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the actor-critic inside a PerturbedWorldModelEnv by rolling
    out until `num_episodes` complete episodes are observed.

    Returns
    -------
    dict with keys: mean_return, std_return
    """
    from torch.distributions.categorical import Categorical

    actor_critic.eval()
    returns = []
    num_envs = perturbed_env.num_envs
    lstm_dim = actor_critic.lstm_dim

    hx = torch.zeros(num_envs, lstm_dim, device=device)
    cx = torch.zeros(num_envs, lstm_dim, device=device)

    obs, _ = perturbed_env.reset()
    ep_returns = torch.zeros(num_envs, device=device)
    episodes_done = 0
    max_steps = num_episodes * 200

    for _ in range(max_steps):
        if episodes_done >= num_episodes:
            break
        with torch.no_grad():
            logits_act, val, (hx, cx) = actor_critic.predict_act_value(obs, (hx, cx))
        act = Categorical(logits=logits_act).sample()

        obs, rew, end, trunc, info = perturbed_env.step(act)
        ep_returns += rew

        done = torch.logical_or(end, trunc)
        if done.any():
            for i in range(num_envs):
                if done[i].item():
                    returns.append(ep_returns[i].item())
                    ep_returns[i] = 0.0
                    episodes_done += 1
            reset_gate = 1 - done.float().unsqueeze(1)
            hx = hx * reset_gate
            cx = cx * reset_gate

    if not returns:
        returns = [0.0]

    return {
        "mean_return": float(np.mean(returns[:num_episodes])),
        "std_return": float(np.std(returns[:num_episodes])),
    }


# ---------------------------------------------------------------------------
# Transfer evaluation helpers (sim-to-real robustness)
# ---------------------------------------------------------------------------

def make_transfer_env(game: str, sticky_prob: float = 0.0, frame_skip: int = 4):
    """
    Build a minimal real Atari environment with modified parameters for
    transfer testing.  Uses standard gymnasium wrappers to apply:
      - sticky_prob: repeat_action_probability (0.0 = none, 0.25 = sticky)
      - frame_skip: effective action repeat / frame skip (default 4)

    Returns a gymnasium env with observations shaped (4, 64, 64) (frame stack
    of 4 greyscale 64x64 frames).
    """
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing, FrameStack

    env = gym.make(
        f"{game}NoFrameskip-v4",
        repeat_action_probability=sticky_prob,
        render_mode=None,
    )
    env = AtariPreprocessing(
        env,
        frame_skip=frame_skip,
        screen_size=64,
        grayscale_obs=True,
        scale_obs=False,  # keep uint8; we normalise manually
    )
    env = FrameStack(env, num_stack=4)
    return env


def evaluate_transfer(
    actor_critic,
    game: str,
    num_episodes: int,
    device: torch.device,
    sticky_prob: float = 0.0,
    obs_noise_std: float = 0.0,
    frame_skip: int = 4,
    condition_label: str = "transfer",
) -> Dict[str, float]:
    """
    Evaluate the actor-critic under a modified real Atari environment
    (transfer / sim-to-real robustness test).

    Supports three independent perturbation modes (can be combined):
      sticky_prob    — 0.25 for Test A (sticky actions)
      obs_noise_std  — 0.05 for Test B (observation noise)
      frame_skip     — 6 for Test C (faster game speed)

    Observations are converted from the gymnasium FrameStack format
    (LazyFrames, shape [4, 64, 64], uint8) to a float32 tensor
    [1, 4, 64, 64] normalised to [0, 1] before passing to actor_critic.

    Returns dict with mean_return, std_return, num_episodes.
    """
    from torch.distributions.categorical import Categorical

    actor_critic.eval()
    returns = []
    lengths = []

    lstm_dim = actor_critic.lstm_dim
    hx = torch.zeros(1, lstm_dim, device=device)
    cx = torch.zeros(1, lstm_dim, device=device)

    print(f"  [Transfer:{condition_label}] Building env "
          f"(sticky={sticky_prob}, noise={obs_noise_std}, frame_skip={frame_skip})…")

    try:
        env = make_transfer_env(game, sticky_prob=sticky_prob, frame_skip=frame_skip)
    except Exception as exc:
        print(f"  [Transfer:{condition_label}] WARNING: could not build transfer env: {exc}")
        print(f"  [Transfer:{condition_label}] Returning zeros.")
        return {"mean_return": 0.0, "std_return": 0.0, "num_episodes": 0}

    ep_return = 0.0
    ep_length = 0
    episodes_done = 0
    max_steps = num_episodes * 5000

    try:
        raw_obs, _ = env.reset()
    except Exception as exc:
        print(f"  [Transfer:{condition_label}] WARNING: env.reset() failed: {exc}")
        env.close()
        return {"mean_return": 0.0, "std_return": 0.0, "num_episodes": 0}

    steps = 0
    while episodes_done < num_episodes and steps < max_steps:
        # Convert LazyFrames / ndarray to tensor float32 in [0,1]
        # FrameStack gives [4, 64, 64] (grayscale). DIAMOND actor_critic expects
        # [1, 3, 64, 64] (last 3 frames as channels). Take frames [-3:, :, :].
        raw_arr = np.array(raw_obs)  # [4, 64, 64]
        raw_arr = raw_arr[-3:, :, :]  # [3, 64, 64]
        obs_tensor = torch.from_numpy(raw_arr).float().unsqueeze(0).to(device) / 255.0  # [1,3,64,64]

        # Test B: add Gaussian noise to normalised observations
        if obs_noise_std > 0.0:
            obs_tensor = obs_tensor + torch.randn_like(obs_tensor) * obs_noise_std
            obs_tensor = obs_tensor.clamp(0.0, 1.0)

        with torch.no_grad():
            logits_act, val, (hx, cx) = actor_critic.predict_act_value(obs_tensor, (hx, cx))
        act = Categorical(logits=logits_act).sample()
        action_int = int(act.item())

        try:
            raw_obs, rew, terminated, truncated, info = env.step(action_int)
        except Exception as exc:
            print(f"  [Transfer:{condition_label}] WARNING: env.step() failed: {exc}")
            break

        ep_return += float(rew)
        ep_length += 1
        steps += 1

        done = terminated or truncated
        if done:
            returns.append(ep_return)
            lengths.append(ep_length)
            episodes_done += 1
            ep_return = 0.0
            ep_length = 0
            hx = hx * 0.0
            cx = cx * 0.0
            try:
                raw_obs, _ = env.reset()
            except Exception:
                break

    env.close()

    if not returns:
        returns = [0.0]
        lengths = [0]

    result = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "num_episodes": len(returns),
    }
    print(
        f"  [Transfer:{condition_label}] "
        f"mean_return={result['mean_return']:.2f} ± {result['std_return']:.2f} "
        f"over {result['num_episodes']} episodes"
    )
    return result


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_comparison_figure(
    results: Dict,
    output_path: Path,
) -> None:
    """
    Bar chart: mean real-env return for each of the 4 agents.
    Error bars = std across evaluation episodes.
    """
    agent_names = ["baseline", "single_perturb", "multi_dream", "adaptive_multi_dream"]
    display_names = [
        "Baseline\n(normal dream)",
        "Single-Perturb\n(mirrored)",
        "Multi-Dream\n(all 5 variants)",
        "Adaptive\n(curriculum)",
    ]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    means = [results["real_env"][a]["mean_return"] for a in agent_names]
    stds = [results["real_env"][a]["std_return"] for a in agent_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(agent_names))
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors,
                  alpha=0.88, edgecolor="black", linewidth=0.8,
                  error_kw={"elinewidth": 2, "ecolor": "dimgray"})

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.set_ylabel("Mean Episode Return (Real Atari)", fontsize=12)
    ax.set_title("Dream Perturbation: Agent Robustness in Real Atari\n(higher = better)", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)

    # Value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(mean + std + 0.5, 1.0),
            f"{mean:.1f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Saved comparison chart: {output_path}")


def generate_robustness_heatmap(
    results: Dict,
    output_path: Path,
    preset_names: List[str],
) -> None:
    """
    Heatmap: rows = agents (4), columns = dream variants (5).
    Cell value = mean return when that agent is evaluated in that variant.
    """
    agent_names = ["baseline", "single_perturb", "multi_dream", "adaptive_multi_dream"]
    display_agent_names = ["Baseline", "Single-Perturb", "Multi-Dream", "Adaptive-Curriculum"]

    matrix = np.zeros((len(agent_names), len(preset_names)))
    for i, agent in enumerate(agent_names):
        for j, preset in enumerate(preset_names):
            val = results.get("dream_variants", {}).get(agent, {}).get(preset, {})
            matrix[i, j] = val.get("mean_return", 0.0) if isinstance(val, dict) else 0.0

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, cmap="YlGn", aspect="auto")

    ax.set_xticks(np.arange(len(preset_names)))
    ax.set_yticks(np.arange(len(agent_names)))
    ax.set_xticklabels([p.replace("_", "\n") for p in preset_names], fontsize=10)
    ax.set_yticklabels(display_agent_names, fontsize=11)

    # Annotate cells
    for i in range(len(agent_names)):
        for j in range(len(preset_names)):
            val = matrix[i, j]
            color = "black" if val < matrix.max() * 0.6 else "white"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Mean Return", shrink=0.85)
    ax.set_title("Agent Robustness Across Dream Variants\n(rows = agents, columns = evaluation environments)",
                 fontsize=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Saved robustness heatmap: {output_path}")


def generate_training_curve_figure(
    training_logs: Dict[str, List[float]],
    output_path: Path,
) -> None:
    """Line plot of training loss (policy entropy) per epoch for all 4 agents."""
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = {
        "baseline": "#4C72B0",
        "single_perturb": "#DD8452",
        "multi_dream": "#55A868",
        "adaptive_multi_dream": "#C44E52",
    }
    labels = {
        "baseline": "Baseline",
        "single_perturb": "Single-Perturb",
        "multi_dream": "Multi-Dream",
        "adaptive_multi_dream": "Adaptive-Curriculum",
    }

    for agent_name, log in training_logs.items():
        if log:
            ax.plot(log, color=colors.get(agent_name, "gray"),
                    label=labels.get(agent_name, agent_name), linewidth=2, alpha=0.85)

    ax.set_xlabel("Training Epoch", fontsize=11)
    ax.set_ylabel("Policy Entropy (bits)", fontsize=11)
    ax.set_title("Training Dynamics: Policy Entropy Across Agents", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Saved training curves: {output_path}")


def generate_transfer_robustness_figure(
    results: Dict,
    output_path: Path,
) -> None:
    """
    Grouped bar chart: 3 groups (sticky_actions, observation_noise, frame_skip_6),
    4 bars each (one per agent).

    Colors:
      Baseline        = #5B89C4
      Single-Perturb  = #E8965D
      Multi-Dream     = #72B77E
      Adaptive        = #C44E52

    A dashed horizontal line per agent shows its standard real-env mean return.
    """
    agent_names = ["baseline", "single_perturb", "multi_dream", "adaptive_multi_dream"]
    display_agent_names = ["Baseline", "Single-Perturb", "Multi-Dream", "Adaptive-Curriculum"]
    agent_colors = {
        "baseline": "#5B89C4",
        "single_perturb": "#E8965D",
        "multi_dream": "#72B77E",
        "adaptive_multi_dream": "#C44E52",
    }

    transfer_conditions = ["sticky_actions", "observation_noise", "frame_skip_6"]
    condition_labels = ["Sticky Actions\n(25% repeat)", "Obs Noise\n(σ=0.05)", "Frame-Skip 6\n(faster)"]

    transfer_data = results.get("transfer", {})
    real_env_data = results.get("real_env", {})

    n_conditions = len(transfer_conditions)
    n_agents = len(agent_names)
    group_width = 0.8
    bar_width = group_width / n_agents

    x = np.arange(n_conditions)

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (agent_name, agent_label, color) in enumerate(
        zip(agent_names, display_agent_names, [agent_colors[a] for a in agent_names])
    ):
        offset = (i - n_agents / 2 + 0.5) * bar_width
        means = []
        stds = []
        for cond in transfer_conditions:
            cond_data = transfer_data.get(agent_name, {}).get(cond, {})
            means.append(cond_data.get("mean_return", 0.0))
            stds.append(cond_data.get("std_return", 0.0))

        bars = ax.bar(
            x + offset, means, bar_width * 0.9,
            yerr=stds, capsize=5,
            color=color, alpha=0.85,
            edgecolor="black", linewidth=0.6,
            error_kw={"elinewidth": 1.5, "ecolor": "dimgray"},
            label=agent_label,
        )

        # Dashed reference line for standard real-env performance
        real_mean = real_env_data.get(agent_name, {}).get("mean_return", None)
        if real_mean is not None:
            for xi in range(n_conditions):
                ax.hlines(
                    real_mean,
                    xi + offset - bar_width * 0.45,
                    xi + offset + bar_width * 0.45,
                    colors=color, linestyles="--", linewidth=1.5, alpha=0.7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, fontsize=11)
    ax.set_ylabel("Mean Episode Return", fontsize=12)
    ax.set_title(
        "Sim-to-Real Transfer: Agent Performance Under Novel Conditions",
        fontsize=13,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-", alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")

    # Add a note explaining dashed lines
    fig.text(
        0.01, 0.01,
        "Dashed lines = each agent's standard real-env performance (reference baseline)",
        fontsize=8, color="gray", va="bottom",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Saved transfer robustness chart: {output_path}")


# ---------------------------------------------------------------------------
# Core training routine for one agent
# ---------------------------------------------------------------------------

PRESET_NAMES = ["normal", "mirrored", "noisy", "shifted_physics", "hard_mode"]


def train_agent(
    agent_copy,
    wm_env_factory,  # callable(actor_critic) -> base WorldModelEnv
    perturbed_env_factory,  # callable(wm_env, preset_name) -> PerturbedWorldModelEnv
    training_mode: str,  # "baseline" | "single_perturb" | "multi_dream" | "adaptive_multi_dream"
    num_epochs: int,
    steps_per_epoch: int,
    device: torch.device,
    args,
) -> Tuple[object, object, List[float]]:
    """
    Train an actor_critic in dreams, return:
      (trained_agent, trained_actor_critic, entropy_log)

    The trained_actor_critic is returned explicitly so the caller can save a
    checkpoint immediately after training completes.

    training_mode
    -------------
    "baseline"             — dream only in "normal" (no perturbation)
    "single_perturb"       — dream only in "mirrored"
    "multi_dream"          — cycle through all 5 variants each epoch
    "adaptive_multi_dream" — adaptive curriculum: sample variants inversely
                             proportional to current performance so the agent
                             spends more time on variants it is BAD at
    """
    import sys
    # Add learning-from-failure/src so we can import PerturbedWorldModelEnv
    lff_src = str(_REPO_ROOT / "src")
    if lff_src not in sys.path:
        sys.path.insert(0, lff_src)
    from perturbed_world_model_env import PerturbedWorldModelEnv, PRESET_CONFIGS
    from models.actor_critic import ActorCriticLossConfig
    from coroutines.env_loop import make_env_loop

    actor_critic = agent_copy.actor_critic

    # Optimiser — AdamW with a small LR appropriate for fine-tuning
    optimizer = torch.optim.AdamW(actor_critic.parameters(), lr=1e-4, weight_decay=1e-5)

    # Determine which presets to cycle through (for non-adaptive modes)
    if training_mode == "baseline":
        cycle_presets = ["normal"]
    elif training_mode == "single_perturb":
        cycle_presets = ["mirrored"]
    elif training_mode == "multi_dream":
        cycle_presets = PRESET_NAMES
    elif training_mode == "adaptive_multi_dream":
        cycle_presets = PRESET_NAMES  # used for initial uniform cycling and eval
    else:
        raise ValueError(f"Unknown training_mode: {training_mode!r}")

    entropy_log: List[float] = []

    print(f"\n{'='*60}")
    print(f"Training agent: {training_mode} ({num_epochs} epochs x {steps_per_epoch} steps)")
    print(f"Dream variants: {cycle_presets}")
    print(f"{'='*60}")

    loss_cfg = ActorCriticLossConfig(
        backup_every=15,
        gamma=0.99,
        lambda_=0.95,
        weight_value_loss=1.0,
        weight_entropy_loss=0.01,
    )

    # Build env_loops lazily (one per preset used)
    env_loops: Dict[str, object] = {}

    # --- Adaptive curriculum state ---
    # variant_scores: running average return per preset (initialised to 0.0)
    variant_scores: Dict[str, float] = {p: 0.0 for p in PRESET_NAMES}
    eval_interval = 10  # re-evaluate every N epochs

    for epoch in range(num_epochs):
        # ------------------------------------------------------------------
        # Pick the dream environment for this epoch
        # ------------------------------------------------------------------
        if training_mode == "adaptive_multi_dream":
            if epoch < eval_interval:
                # First eval_interval epochs: cycle uniformly like multi_dream
                preset_name = cycle_presets[epoch % len(cycle_presets)]
            else:
                # Sample with probability inversely proportional to score
                epsilon = 0.1
                weights = [1.0 / (variant_scores[p] + epsilon) for p in PRESET_NAMES]
                total_w = sum(weights)
                weights_norm = [w / total_w for w in weights]
                preset_name = random.choices(PRESET_NAMES, weights=weights_norm, k=1)[0]
            print(f"  [adaptive] Epoch {epoch+1}: selected preset={preset_name}")
        else:
            # Round-robin for baseline / single_perturb / multi_dream
            preset_name = cycle_presets[epoch % len(cycle_presets)]

        # Build or reuse the env_loop for this preset
        if preset_name not in env_loops:
            base_wm_env = wm_env_factory(actor_critic)
            perturbed = perturbed_env_factory(base_wm_env, preset_name)
            actor_critic_tmp = actor_critic  # reference for env_loop
            env_loop = make_env_loop(perturbed, actor_critic_tmp)
            env_loops[preset_name] = (env_loop, perturbed)

        env_loop, perturbed = env_loops[preset_name]

        # Temporarily attach the env_loop so actor_critic.forward() works
        _orig_env_loop = actor_critic.env_loop
        actor_critic.env_loop = env_loop
        actor_critic.loss_cfg = loss_cfg

        actor_critic.train()
        metrics = train_actor_critic_one_epoch(
            actor_critic=actor_critic,
            rl_env=perturbed,
            optimizer=optimizer,
            grad_acc_steps=max(1, steps_per_epoch // 15),
            backup_every=15,
        )

        # Restore
        actor_critic.env_loop = _orig_env_loop

        entropy_log.append(metrics.get("policy_entropy", 0.0))

        if (epoch + 1) % max(1, num_epochs // 5) == 0:
            print(
                f"  Epoch {epoch+1:3d}/{num_epochs} | preset={preset_name:20s} | "
                f"loss={metrics['loss_total']:.4f} | "
                f"entropy={metrics['policy_entropy']:.3f} bits"
            )

        # ------------------------------------------------------------------
        # Adaptive curriculum: re-evaluate every eval_interval epochs
        # ------------------------------------------------------------------
        if training_mode == "adaptive_multi_dream" and (epoch + 1) % eval_interval == 0:
            print(f"  [adaptive] Running quick evaluation after epoch {epoch+1}…")
            for preset_eval in PRESET_NAMES:
                # Build a fresh eval env for this preset
                base_wm_eval = wm_env_factory(actor_critic)
                perturbed_eval = perturbed_env_factory(base_wm_eval, preset_eval)
                eval_result = evaluate_in_dream_variant(
                    actor_critic=actor_critic,
                    perturbed_env=perturbed_eval,
                    num_episodes=3,
                    device=device,
                )
                # Update running average (simple exponential smoothing, alpha=0.5)
                old_score = variant_scores[preset_eval]
                new_score = eval_result["mean_return"]
                variant_scores[preset_eval] = 0.5 * old_score + 0.5 * new_score
                print(
                    f"    variant={preset_eval:20s} | "
                    f"eval_return={new_score:.2f} | "
                    f"running_avg={variant_scores[preset_eval]:.2f}"
                )

    # Return the agent, the actor_critic module, and the entropy log.
    # Caller uses actor_critic to save checkpoint immediately.
    return agent_copy, actor_critic, entropy_log


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Quick mode overrides
    if args.fast:
        args.num_epochs = 10
        args.train_steps_per_epoch = 50
        args.eval_episodes = 5
        args.init_collect_steps = 200
        print("[Fast mode] Reduced epochs/steps for quick testing")

    # Set up paths
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Apply patches first (before any DIAMOND imports)
    apply_patches()

    # Set up DIAMOND imports
    setup_diamond_imports(args.diamond_root)

    # Now import DIAMOND modules
    _patch_batch_sampler()

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from agent import Agent
    from coroutines.collector import make_collector, NumToCollect
    from data import BatchSampler, collate_segments_to_batch, Dataset
    from envs import make_atari_env, WorldModelEnv, WorldModelEnvConfig
    from models.diffusion import DiffusionSamplerConfig
    import sys
    lff_src = str(_REPO_ROOT / "src")
    if lff_src not in sys.path:
        sys.path.insert(0, lff_src)
    from perturbed_world_model_env import PerturbedWorldModelEnv, PRESET_CONFIGS

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Using device: {device}")

    # -----------------------------------------------------------------------
    # 1. Load checkpoint + configs
    # -----------------------------------------------------------------------
    if args.checkpoint:
        path_ckpt = Path(args.checkpoint)
        print(f"[Checkpoint] Using provided checkpoint: {path_ckpt}")
        # Load configs from HuggingFace (needed to reconstruct agent architecture)
        agent_cfg_path, env_cfg_path = download_pretrained_configs()
    else:
        path_ckpt = download_pretrained_checkpoint(args.game)
        agent_cfg_path, env_cfg_path = download_pretrained_configs()

    agent_cfg_raw = OmegaConf.load(agent_cfg_path)
    env_cfg_raw = OmegaConf.load(env_cfg_path)

    # Resolve game name
    game_name = args.game
    env_cfg_raw.train.id = f"{game_name}NoFrameskip-v4"
    env_cfg_raw.test.id = f"{game_name}NoFrameskip-v4"

    # -----------------------------------------------------------------------
    # 2a. Resolve cross-config interpolations in agent config
    # -----------------------------------------------------------------------
    # DIAMOND's Hydra config uses interpolations like
    #   ${agent.denoiser.inner_model.img_channels}  and  ${env.train.size}
    # These only resolve inside the full Hydra config tree.  Since we load
    # agent.yaml and env.yaml separately, we must resolve them manually.
    img_channels = int(agent_cfg_raw.denoiser.inner_model.img_channels)   # 3
    img_size = int(env_cfg_raw.train.size)                                # 64

    # Patch rew_end_model interpolations
    agent_cfg_raw.rew_end_model.img_channels = img_channels
    agent_cfg_raw.rew_end_model.img_size = img_size
    # Patch actor_critic interpolations
    agent_cfg_raw.actor_critic.img_channels = img_channels
    agent_cfg_raw.actor_critic.img_size = img_size
    print(f"[Config] Resolved img_channels={img_channels}, img_size={img_size}")

    # -----------------------------------------------------------------------
    # 2b. Build real Atari envs (1 env for eval, num_envs for data collection)
    # -----------------------------------------------------------------------
    print("[Envs] Creating real Atari environments…")
    train_env = make_atari_env(num_envs=1, device=device, **env_cfg_raw.train)
    test_env = make_atari_env(num_envs=1, device=device, **env_cfg_raw.test)
    num_actions = int(test_env.num_actions)
    print(f"[Envs] num_actions = {num_actions}")

    # -----------------------------------------------------------------------
    # 3. Load pretrained DIAMOND agent
    # -----------------------------------------------------------------------
    print("[Agent] Loading DIAMOND agent…")
    agent_cfg = instantiate(agent_cfg_raw, num_actions=num_actions)
    agent = Agent(agent_cfg).to(device).eval()
    agent.load(path_ckpt)
    print("[Agent] Checkpoint loaded successfully")

    # -----------------------------------------------------------------------
    # 4. Collect initial dataset for world-model burn-in
    # -----------------------------------------------------------------------
    dataset_dir = output_dir / "dataset" / "train"
    dataset = Dataset(dataset_dir, "train_dataset", cache_in_ram=False)
    dataset.load_from_default_path()

    if dataset.num_steps < args.init_collect_steps:
        collect_initial_dataset(
            real_env=train_env,
            actor_critic=agent.actor_critic,
            dataset=dataset,
            num_steps=args.init_collect_steps,
            device=device,
        )
        dataset.save_to_default_path()

    # -----------------------------------------------------------------------
    # 5. Factory functions for world-model envs
    # -----------------------------------------------------------------------
    num_steps_conditioning = agent_cfg_raw.denoiser.inner_model.num_steps_conditioning

    def wm_env_factory(actor_critic_ref) -> WorldModelEnv:
        """Create a fresh WorldModelEnv (new DataLoader, new burn-in state)."""
        return build_world_model_env(
            agent=agent,
            dataset=dataset,
            batch_size=args.batch_size,
            horizon=args.wm_horizon,
            num_steps_conditioning=num_steps_conditioning,
            device=device,
        )

    def perturbed_env_factory(wm_env: WorldModelEnv, preset_name: str) -> PerturbedWorldModelEnv:
        return PerturbedWorldModelEnv(wm_env, perturbation_config=preset_name)

    # -----------------------------------------------------------------------
    # 6. Train 4 agents
    # -----------------------------------------------------------------------
    # Deep-copy the pretrained agent for each training run so they all start
    # from the same pretrained weights.
    results: Dict = {
        "config": {
            "num_epochs": args.num_epochs,
            "train_steps_per_epoch": args.train_steps_per_epoch,
            "eval_episodes": args.eval_episodes,
            "wm_horizon": args.wm_horizon,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": str(device),
            "checkpoint": str(path_ckpt),
            "game": game_name,
        },
        "real_env": {},
        "dream_variants": {},
        "training_logs": {},
        "transfer": {},
    }

    training_jobs = [
        ("baseline", "Baseline (normal dream only)"),
        ("single_perturb", "Single-Perturb (mirrored dream)"),
        ("multi_dream", "Multi-Dream (all 5 variants)"),
        ("adaptive_multi_dream", "Adaptive-Curriculum (all 5 variants, adaptive)"),
    ]

    # Human-readable display names used in summary tables
    agent_display_names = {
        "baseline": "Baseline",
        "single_perturb": "Single-Perturb",
        "multi_dream": "Multi-Dream",
        "adaptive_multi_dream": "Adaptive-Curriculum",
    }

    trained_agents = {}

    # Checkpoint directory (created once, reused for all agents)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for training_mode, description in training_jobs:
        print(f"\n{'#'*60}")
        print(f"# Training: {description}")
        print(f"{'#'*60}")

        agent_copy = copy.deepcopy(agent)
        trained_agent, actor_critic, entropy_log = train_agent(
            agent_copy=agent_copy,
            wm_env_factory=wm_env_factory,
            perturbed_env_factory=perturbed_env_factory,
            training_mode=training_mode,
            num_epochs=args.num_epochs,
            steps_per_epoch=args.train_steps_per_epoch,
            device=device,
            args=args,
        )

        # -------------------------------------------------------------------
        # Save actor_critic checkpoint immediately after training
        # -------------------------------------------------------------------
        agent_name = training_mode  # e.g. "baseline", "single_perturb", …
        ckpt_path = ckpt_dir / f"{agent_name}_actor_critic.pt"
        torch.save(actor_critic.state_dict(), ckpt_path)
        print(f"[Checkpoint] Saved {agent_name} to {ckpt_path}")

        trained_agents[training_mode] = trained_agent
        results["training_logs"][training_mode] = entropy_log

    # -----------------------------------------------------------------------
    # 7. Evaluate all 4 agents in the REAL Atari environment
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Evaluating agents in REAL Atari environment…")
    print(f"{'='*60}")

    for training_mode, _ in training_jobs:
        print(f"  Evaluating: {training_mode}")
        eval_results = evaluate_in_real_env(
            actor_critic=trained_agents[training_mode].actor_critic,
            real_env=test_env,
            num_episodes=args.eval_episodes,
            device=device,
        )
        results["real_env"][training_mode] = eval_results
        print(
            f"    mean_return={eval_results['mean_return']:.2f} ± {eval_results['std_return']:.2f} "
            f"over {eval_results['num_episodes']} episodes"
        )

    # -----------------------------------------------------------------------
    # 8. Evaluate all 4 agents in HELD-OUT dream variants
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Evaluating agents in dream variant environments…")
    print(f"{'='*60}")

    for training_mode, _ in training_jobs:
        results["dream_variants"][training_mode] = {}
        for preset_name in PRESET_NAMES:
            print(f"  Agent={training_mode:25s} | Variant={preset_name}")
            base_wm = wm_env_factory(trained_agents[training_mode].actor_critic)
            perturbed = perturbed_env_factory(base_wm, preset_name)
            dream_eval = evaluate_in_dream_variant(
                actor_critic=trained_agents[training_mode].actor_critic,
                perturbed_env=perturbed,
                num_episodes=max(3, args.eval_episodes // 2),
                device=device,
            )
            results["dream_variants"][training_mode][preset_name] = dream_eval
            print(
                f"    mean_return={dream_eval['mean_return']:.2f} ± {dream_eval['std_return']:.2f}"
            )

    # -----------------------------------------------------------------------
    # 9. Transfer Tests (sim-to-real robustness)
    #    Each trained agent is evaluated under 3 novel real-env conditions
    #    that were NOT present during training.
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("############################################################")
    print("# Transfer Tests (sim-to-real robustness)")
    print("############################################################")
    print(f"{'='*60}")
    print(f"  Running {args.eval_episodes} episodes per agent per condition…")

    for training_mode, _ in training_jobs:
        actor_critic = trained_agents[training_mode].actor_critic

        print(f"\n  --- Agent: {training_mode} ---")

        # Test A: Sticky Actions (repeat_action_probability=0.25)
        sticky_result = evaluate_transfer(
            actor_critic=actor_critic,
            game=game_name,
            num_episodes=args.eval_episodes,
            device=device,
            sticky_prob=0.25,
            obs_noise_std=0.0,
            frame_skip=4,
            condition_label="sticky_actions",
        )

        # Test B: Observation Noise (Gaussian std=0.05 on normalised obs)
        noise_result = evaluate_transfer(
            actor_critic=actor_critic,
            game=game_name,
            num_episodes=args.eval_episodes,
            device=device,
            sticky_prob=0.0,
            obs_noise_std=0.05,
            frame_skip=4,
            condition_label="observation_noise",
        )

        # Test C: Frame-Skip 6 (faster game speed, non-standard)
        frameskip_result = evaluate_transfer(
            actor_critic=actor_critic,
            game=game_name,
            num_episodes=args.eval_episodes,
            device=device,
            sticky_prob=0.0,
            obs_noise_std=0.0,
            frame_skip=6,
            condition_label="frame_skip_6",
        )

        results["transfer"][training_mode] = {
            "sticky_actions": {
                "mean_return": sticky_result["mean_return"],
                "std_return": sticky_result["std_return"],
            },
            "observation_noise": {
                "mean_return": noise_result["mean_return"],
                "std_return": noise_result["std_return"],
            },
            "frame_skip_6": {
                "mean_return": frameskip_result["mean_return"],
                "std_return": frameskip_result["std_return"],
            },
        }

    # -----------------------------------------------------------------------
    # 10. Save results JSON
    # -----------------------------------------------------------------------
    results_path = output_dir / "dream_perturbation_results_v3.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results] Saved to: {results_path}")

    # -----------------------------------------------------------------------
    # 11. Generate figures
    # -----------------------------------------------------------------------
    generate_comparison_figure(
        results=results,
        output_path=figures_dir / "dream_perturbation_comparison_v3.png",
    )

    generate_robustness_heatmap(
        results=results,
        output_path=figures_dir / "dream_robustness_v3.png",
        preset_names=PRESET_NAMES,
    )

    generate_training_curve_figure(
        training_logs=results["training_logs"],
        output_path=figures_dir / "training_curves_v3.png",
    )

    generate_transfer_robustness_figure(
        results=results,
        output_path=figures_dir / "transfer_robustness.png",
    )

    # -----------------------------------------------------------------------
    # 12. Print summary tables
    # -----------------------------------------------------------------------
    agent_names = ["baseline", "single_perturb", "multi_dream", "adaptive_multi_dream"]

    print(f"\n{'='*60}")
    print("SUMMARY — Mean Return in Real Atari Environment")
    print(f"{'='*60}")
    print(f"{'Agent':<40} {'Mean Return':>12} {'Std':>8}")
    print("-" * 62)
    for training_mode, desc in training_jobs:
        r = results["real_env"][training_mode]
        print(f"{desc:<40} {r['mean_return']:>12.2f} {r['std_return']:>8.2f}")

    # Robustness: average return across all dream variants
    print(f"\n{'='*60}")
    print("SUMMARY — Mean Dream-Variant Return (held-out robustness)")
    print(f"{'='*60}")
    print(f"{'Agent':<40} {'Mean over variants':>20}")
    print("-" * 62)
    for training_mode, desc in training_jobs:
        variant_returns = [
            results["dream_variants"][training_mode][p]["mean_return"]
            for p in PRESET_NAMES
        ]
        print(f"{desc:<40} {np.mean(variant_returns):>20.2f}")

    # Transfer summary
    print(f"\n{'='*62}")
    print("SUMMARY — Transfer Tests (novel conditions, not seen in training)")
    print(f"{'='*62}")
    header = f"{'Agent':<28} {'Sticky Actions':>16} {'Obs Noise':>12} {'Frame-Skip 6':>14}"
    print(header)
    print("-" * 72)
    for training_mode in agent_names:
        disp = agent_display_names[training_mode]
        t = results["transfer"].get(training_mode, {})
        sticky_mean = t.get("sticky_actions", {}).get("mean_return", float("nan"))
        noise_mean = t.get("observation_noise", {}).get("mean_return", float("nan"))
        fs_mean = t.get("frame_skip_6", {}).get("mean_return", float("nan"))
        print(
            f"{disp:<28} {sticky_mean:>16.2f} {noise_mean:>12.2f} {fs_mean:>14.2f}"
        )

    print(f"\n[Done] All results and figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
