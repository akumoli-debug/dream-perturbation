"""
perturbed_world_model_env.py
============================
PerturbedWorldModelEnv — wraps DIAMOND's WorldModelEnv to inject
controllable perturbations during RL training.

The core idea: by training an actor-critic across many *dream variants*,
the agent learns behaviours that are more robust to real-world distribution
shifts.  Perturbations are applied AFTER the denoiser generates each frame,
so the world-model weights themselves are never touched.

Perturbation categories
-----------------------
1. Frame perturbations  — visual transforms on the generated observation
2. Action remapping     — reorder / suppress action indices before passing to env
3. Reward perturbation  — scale or delay the scalar reward
4. Physics perturbation — directional bias added to the diffusion trajectory

Preset configs
--------------
  "normal"          — no perturbations (baseline)
  "mirrored"        — horizontal flip + left/right action swap
  "noisy"           — Gaussian frame noise + brightness jitter
  "shifted_physics" — directional diffusion noise bias
  "hard_mode"       — everything at once (all perturbations combined)

Usage
-----
    from perturbed_world_model_env import PerturbedWorldModelEnv, PRESET_CONFIGS

    wm_env = WorldModelEnv(denoiser, rew_end_model, data_loader, cfg)
    perturbed = PerturbedWorldModelEnv(wm_env, perturbation_config=PRESET_CONFIGS["mirrored"])

The PerturbedWorldModelEnv exposes the same interface as WorldModelEnv
(reset, step, num_envs, device) so it is a drop-in replacement for the
actor_critic training loop.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Typing aliases (mirroring WorldModelEnv)
# ---------------------------------------------------------------------------
ResetOutput = Tuple[torch.FloatTensor, Dict[str, Any]]
StepOutput = Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]


# ---------------------------------------------------------------------------
# Perturbation config dataclass
# ---------------------------------------------------------------------------

@dataclass
class PerturbationConfig:
    """
    Describes which perturbations to apply and with what parameters.

    All fields default to "disabled" (no-op) so you can turn individual
    perturbations on independently.
    """

    # --- Frame perturbations ------------------------------------------------
    horizontal_flip: bool = False
    """Mirror every generated frame left-to-right."""

    brightness_shift: float = 0.0
    """Additive brightness offset in [-1, 1] pixel space.  0 = disabled."""

    contrast_scale: float = 1.0
    """Multiplicative contrast factor.  1.0 = no change.  0.8 → lower contrast."""

    spatial_shift_px: int = 0
    """Translate the frame by this many pixels (randomly chosen direction each step)."""

    gaussian_noise_std: float = 0.0
    """Std-dev of zero-mean Gaussian noise added to each frame.  0 = disabled."""

    # --- Action remapping ---------------------------------------------------
    swap_left_right: bool = False
    """Swap the Atari LEFT and RIGHT action indices (actions 3 and 4 in most games)."""

    action_dropout_prob: float = 0.0
    """Probability of replacing any action with NOOP (action 0) before stepping."""

    # --- Reward perturbation ------------------------------------------------
    reward_scale: float = 1.0
    """Multiply every raw reward by this factor.  1.0 = no change."""

    reward_delay_steps: int = 0
    """Hold rewards in a FIFO buffer and emit them N steps later.  0 = disabled."""

    # --- Physics perturbation (diffusion-level) ----------------------------
    physics_bias_strength: float = 0.0
    """
    Magnitude of a fixed directional vector added to the denoiser output at
    each sampling step.  Simulates altered 'gravity' or 'friction' dynamics.
    0.0 = disabled.
    """

    physics_bias_direction: str = "up"
    """
    Direction of the physics bias: one of "up", "down", "left", "right".
    Implemented as a pixel-space gradient on the generated frame.
    """

    # --- Metadata -----------------------------------------------------------
    name: str = "custom"
    """Human-readable name for logging / figure labels."""


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

PRESET_CONFIGS: Dict[str, PerturbationConfig] = {
    # ---- Baseline: pristine dream environment ----
    "normal": PerturbationConfig(
        name="normal",
    ),

    # ---- Mirrored: flipped game + swapped controls ----
    # The agent must learn to play a horizontally mirrored version of Breakout.
    "mirrored": PerturbationConfig(
        name="mirrored",
        horizontal_flip=True,
        swap_left_right=True,
    ),

    # ---- Noisy: corrupted observations ----
    # Tests robustness to partial observability and sensor noise.
    "noisy": PerturbationConfig(
        name="noisy",
        gaussian_noise_std=0.08,
        brightness_shift=0.05,
        contrast_scale=0.90,
    ),

    # ---- Shifted physics: altered diffusion dynamics ----
    # Adds a directional bias to the denoised frame, which propagates through
    # the obs_buffer into future denoising steps, mimicking changed 'physics'.
    "shifted_physics": PerturbationConfig(
        name="shifted_physics",
        physics_bias_strength=0.04,
        physics_bias_direction="down",
        reward_scale=0.9,
    ),

    # ---- Hard mode: everything combined ----
    # The most challenging dream variant; combines all perturbation types.
    "hard_mode": PerturbationConfig(
        name="hard_mode",
        horizontal_flip=True,
        swap_left_right=True,
        gaussian_noise_std=0.06,
        brightness_shift=0.03,
        contrast_scale=0.85,
        spatial_shift_px=3,
        action_dropout_prob=0.05,
        reward_scale=0.75,
        reward_delay_steps=1,
        physics_bias_strength=0.03,
        physics_bias_direction="right",
    ),
}


# ---------------------------------------------------------------------------
# Utility: build a pixel-space directional gradient tensor
# ---------------------------------------------------------------------------

def _make_directional_gradient(
    direction: str,
    batch: int,
    channels: int,
    height: int,
    width: int,
    device: torch.device,
) -> Tensor:
    """
    Returns a [B, C, H, W] tensor whose values form a normalised linear ramp
    in the specified direction.  This is added to the denoised frame to create
    a spatial bias that feels like a directional force on the game dynamics.

    The gradient is normalised to have unit max-abs value so that
    ``physics_bias_strength`` controls the actual perturbation magnitude.
    """
    if direction == "up":
        # High values at top, low at bottom
        ramp = torch.linspace(1.0, -1.0, height, device=device).view(1, 1, height, 1)
        ramp = ramp.expand(batch, channels, height, width)
    elif direction == "down":
        ramp = torch.linspace(-1.0, 1.0, height, device=device).view(1, 1, height, 1)
        ramp = ramp.expand(batch, channels, height, width)
    elif direction == "left":
        ramp = torch.linspace(1.0, -1.0, width, device=device).view(1, 1, 1, width)
        ramp = ramp.expand(batch, channels, height, width)
    elif direction == "right":
        ramp = torch.linspace(-1.0, 1.0, width, device=device).view(1, 1, 1, width)
        ramp = ramp.expand(batch, channels, height, width)
    else:
        raise ValueError(f"Unknown physics_bias_direction: {direction!r}. "
                         f"Choose from 'up', 'down', 'left', 'right'.")
    return ramp.float()


# ---------------------------------------------------------------------------
# Action remap helpers
# ---------------------------------------------------------------------------

# Standard Atari action indices for most games (ALE default):
#   0 = NOOP, 1 = FIRE, 2 = UP, 3 = RIGHT, 4 = LEFT, 5 = DOWN
_ATARI_LEFT_ACTION = 4
_ATARI_RIGHT_ACTION = 3


def _build_action_remap(
    num_actions: int,
    swap_left_right: bool,
) -> Optional[Tensor]:
    """
    Returns a 1-D LongTensor mapping old action index -> new action index,
    or None if no remapping is needed.
    """
    if not swap_left_right:
        return None
    remap = torch.arange(num_actions, dtype=torch.long)
    # Swap LEFT and RIGHT only if both indices exist
    if _ATARI_LEFT_ACTION < num_actions and _ATARI_RIGHT_ACTION < num_actions:
        remap[_ATARI_LEFT_ACTION] = _ATARI_RIGHT_ACTION
        remap[_ATARI_RIGHT_ACTION] = _ATARI_LEFT_ACTION
    return remap


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PerturbedWorldModelEnv:
    """
    Drop-in replacement for DIAMOND's WorldModelEnv that applies a
    configurable set of perturbations during RL roll-outs.

    Parameters
    ----------
    world_model_env : WorldModelEnv
        The underlying (unperturbed) DIAMOND world-model environment.
    perturbation_config : PerturbationConfig | str
        Either a PerturbationConfig instance or a string key into
        PRESET_CONFIGS (e.g. "mirrored", "noisy", "shifted_physics").
    seed : int, optional
        RNG seed for reproducibility of stochastic perturbations.
    """

    def __init__(
        self,
        world_model_env,  # WorldModelEnv — avoid hard import for portability
        perturbation_config: "PerturbationConfig | str" = "normal",
        seed: int = 42,
    ) -> None:
        self._env = world_model_env

        # Resolve string shorthand
        if isinstance(perturbation_config, str):
            if perturbation_config not in PRESET_CONFIGS:
                raise ValueError(
                    f"Unknown preset {perturbation_config!r}. "
                    f"Choose from: {list(PRESET_CONFIGS)}"
                )
            perturbation_config = PRESET_CONFIGS[perturbation_config]
        self.cfg = perturbation_config

        # Seeded RNG for reproducible stochastic perturbations
        self._rng = random.Random(seed)

        # Lazy-initialised action remap tensor (needs num_actions)
        self._action_remap: Optional[Tensor] = None
        self._num_actions_cached: Optional[int] = None

        # Reward delay buffer: one deque per parallel environment
        self._reward_buffer: Optional[List[deque]] = None

    # ------------------------------------------------------------------
    # Env interface properties
    # ------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> torch.device:
        return self._env.device

    # num_actions is read by make_env_loop
    @property
    def num_actions(self) -> int:
        return self._env.num_envs  # fallback; real value set when available

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> ResetOutput:
        obs, info = self._env.reset(**kwargs)

        # Initialise reward delay buffer after we know num_envs
        if self.cfg.reward_delay_steps > 0 and self._reward_buffer is None:
            self._reward_buffer = [
                deque([0.0] * self.cfg.reward_delay_steps,
                      maxlen=self.cfg.reward_delay_steps)
                for _ in range(self.num_envs)
            ]

        return self._perturb_obs(obs), info

    def reset_dead(self, dead: torch.BoolTensor) -> None:
        """Forward reset_dead to the underlying env (called internally by step)."""
        self._env.reset_dead(dead)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, act: torch.LongTensor) -> StepOutput:
        """
        Apply action perturbations, delegate to the underlying env,
        then apply frame / reward / physics perturbations to the output.
        """
        act = self._perturb_action(act)

        # Call underlying world-model step
        obs, rew, end, trunc, info = self._env.step(act)

        # Apply frame perturbation to the returned observation
        obs = self._perturb_obs(obs)

        # Apply physics bias: add directional gradient to frame, clamp to [-1,1]
        if self.cfg.physics_bias_strength > 0.0:
            obs = self._apply_physics_bias(obs)

        # Apply reward perturbation
        rew = self._perturb_reward(rew)

        # Perturb "final_observation" in info if present (used by env_loop for
        # value bootstrapping on episode end)
        if "final_observation" in info:
            info["final_observation"] = self._perturb_obs(info["final_observation"])
            if self.cfg.physics_bias_strength > 0.0:
                info["final_observation"] = self._apply_physics_bias(
                    info["final_observation"]
                )

        return obs, rew, end, trunc, info

    # ------------------------------------------------------------------
    # predict_next_obs / predict_rew_end — forwarded for compile compat
    # ------------------------------------------------------------------

    def predict_next_obs(self):
        return self._env.predict_next_obs()

    def predict_rew_end(self, *args, **kwargs):
        return self._env.predict_rew_end(*args, **kwargs)

    # ------------------------------------------------------------------
    # Frame perturbation
    # ------------------------------------------------------------------

    def _perturb_obs(self, obs: Tensor) -> Tensor:
        """Apply all enabled frame-level perturbations."""
        # Horizontal flip
        if self.cfg.horizontal_flip:
            obs = obs.flip(dims=[-1])  # flip width dimension

        # Brightness shift (additive)
        if self.cfg.brightness_shift != 0.0:
            obs = obs + self.cfg.brightness_shift

        # Contrast scaling (around the midpoint 0.0 in [-1,1] space)
        if self.cfg.contrast_scale != 1.0:
            obs = obs * self.cfg.contrast_scale

        # Spatial translation
        if self.cfg.spatial_shift_px > 0:
            obs = self._apply_spatial_shift(obs)

        # Additive Gaussian noise
        if self.cfg.gaussian_noise_std > 0.0:
            noise = torch.randn_like(obs) * self.cfg.gaussian_noise_std
            obs = obs + noise

        # Clamp back to valid pixel range
        obs = obs.clamp(-1.0, 1.0)
        return obs

    def _apply_spatial_shift(self, obs: Tensor) -> Tensor:
        """
        Translate the frame by up to ``spatial_shift_px`` pixels in a random
        direction using reflect padding so there are no black borders.
        """
        px = self.cfg.spatial_shift_px
        shift_h = self._rng.randint(-px, px)
        shift_w = self._rng.randint(-px, px)

        if shift_h == 0 and shift_w == 0:
            return obs

        # Use F.pad with reflect mode then crop
        # obs: [B, C, H, W]
        pad_h_top = max(0, shift_h)
        pad_h_bot = max(0, -shift_h)
        pad_w_left = max(0, shift_w)
        pad_w_right = max(0, -shift_w)

        # reflect padding
        obs_padded = F.pad(obs, (pad_w_left, pad_w_right, pad_h_top, pad_h_bot), mode="reflect")

        # Crop back to original size
        h, w = obs.shape[-2], obs.shape[-1]
        start_h = pad_h_bot  # offset from padded top
        start_w = pad_w_right
        obs = obs_padded[..., start_h:start_h + h, start_w:start_w + w]
        return obs

    # ------------------------------------------------------------------
    # Physics perturbation
    # ------------------------------------------------------------------

    def _apply_physics_bias(self, obs: Tensor) -> Tensor:
        """
        Add a fixed directional ramp to the generated frame.  Because the
        obs feeds back into the next denoising step via obs_buffer, this
        creates a persistent spatial bias that manifests as altered dynamics
        (the "gravity / friction" analogy).
        """
        b, c, h, w = obs.shape
        gradient = _make_directional_gradient(
            self.cfg.physics_bias_direction, b, c, h, w, obs.device
        )
        return (obs + self.cfg.physics_bias_strength * gradient).clamp(-1.0, 1.0)

    # ------------------------------------------------------------------
    # Action perturbation
    # ------------------------------------------------------------------

    def _perturb_action(self, act: torch.LongTensor) -> torch.LongTensor:
        """Remap and/or drop out actions before forwarding to the world model."""
        # Action dropout: replace random actions with NOOP (0)
        if self.cfg.action_dropout_prob > 0.0:
            mask = torch.rand(act.shape, device=act.device) < self.cfg.action_dropout_prob
            act = act.clone()
            act[mask] = 0  # NOOP

        # Left/right swap
        if self.cfg.swap_left_right:
            act = self._get_action_remap(act.max().item() + 1)[act]

        return act

    def _get_action_remap(self, num_actions: int) -> Tensor:
        """
        Lazily build (and cache) the action remap tensor on the correct device.
        """
        if (
            self._action_remap is None
            or self._num_actions_cached != num_actions
        ):
            self._action_remap = _build_action_remap(
                num_actions, self.cfg.swap_left_right
            )
            if self._action_remap is not None:
                self._action_remap = self._action_remap.to(self.device)
            self._num_actions_cached = num_actions
        return self._action_remap

    # ------------------------------------------------------------------
    # Reward perturbation
    # ------------------------------------------------------------------

    def _perturb_reward(self, rew: Tensor) -> Tensor:
        """Scale and/or delay scalar rewards."""
        # Scale
        if self.cfg.reward_scale != 1.0:
            rew = rew * self.cfg.reward_scale

        # Delay: push current reward into buffer, pop the oldest
        if self.cfg.reward_delay_steps > 0 and self._reward_buffer is not None:
            delayed_rew = rew.clone()
            for i in range(self.num_envs):
                r_i = rew[i].item()
                self._reward_buffer[i].append(r_i)
                delayed_rew[i] = self._reward_buffer[i][0]
            rew = delayed_rew

        return rew

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PerturbedWorldModelEnv("
            f"preset={self.cfg.name!r}, "
            f"num_envs={self.num_envs}, "
            f"device={self.device})"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_perturbed_envs(
    world_model_env,
    presets: Optional[List[str]] = None,
) -> Dict[str, PerturbedWorldModelEnv]:
    """
    Create one PerturbedWorldModelEnv per preset from a single base
    WorldModelEnv.

    Parameters
    ----------
    world_model_env : WorldModelEnv
        The base DIAMOND world-model environment (already initialised).
    presets : list of str, optional
        Subset of preset names to build. Defaults to all five presets.

    Returns
    -------
    dict mapping preset name -> PerturbedWorldModelEnv
    """
    if presets is None:
        presets = list(PRESET_CONFIGS.keys())
    return {
        name: PerturbedWorldModelEnv(world_model_env, perturbation_config=name)
        for name in presets
    }
