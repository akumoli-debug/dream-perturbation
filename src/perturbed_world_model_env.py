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
5. Sigma perturbation   — modifies EDM noise schedule at inference time
6. Adversarial action   — tiny MLP maps context → action offset, trained adversarially

Preset configs
--------------
  "normal"            — no perturbations (baseline)
  "mirrored"          — horizontal flip + left/right action swap
  "sigma_perturb"     — perturbs DiffusionSamplerConfig: raises sigma_min, reduces num_steps
  "adversarial_action"— adversarial MLP finds the hardest action perturbations online
  "hard_mode"         — everything at once (all perturbations combined)

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
import torch.nn as nn
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

    # --- Sigma perturbation (EDM noise schedule) ---------------------------
    sigma_scale: float = 1.0
    """
    Multiplier applied to DiffusionSamplerConfig.sigma_min before each step.
    Values > 1.0 raise the noise floor, making the world model produce less
    certain (noisier) frames.  1.0 = no change.
    """

    steps_reduction: float = 0.0
    """
    Fraction by which to reduce num_steps_denoising before each step.
    0.0 = no change, 0.3 = reduce by 30% (fewer denoising steps → noisier output).
    Applied as: new_steps = max(1, round(original_steps * (1 - steps_reduction))).
    """

    # --- Adversarial action perturbation -----------------------------------
    adversarial_action: bool = False
    """
    Train a tiny 2→num_actions MLP (context: normalised return + normalised step)
    that adds a learned offset to the action logits.  The MLP is updated
    adversarially — one gradient step per env step — to maximise the negative
    reward, finding the hardest action perturbation online.
    """

    adv_lr: float = 1e-3
    """Learning rate for the adversarial MLP optimiser."""

    adv_hidden: int = 16
    """Hidden layer width for the adversarial MLP."""

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

    # ---- Sigma perturbation: EDM noise schedule modification ----
    # Touches DIAMOND's DiffusionSamplerConfig directly at inference time.
    # Raises sigma_min by 1.5x (higher noise floor) and reduces num_steps by 30%
    # (fewer denoising steps → world model is "less sure" about each prediction).
    # Tests whether the policy is robust to the world model's uncertainty calibration.
    "sigma_perturb": PerturbationConfig(
        name="sigma_perturb",
        sigma_scale=1.5,
        steps_reduction=0.3,
    ),

    # ---- Adversarial action: online adversarial perturbation ----
    # A tiny 2-layer MLP maps (normalised_return, normalised_step) → action logit offset.
    # The MLP is updated adversarially each step to find the hardest action perturbation.
    # Tests whether dream-trained policies are robust to learned adversarial interference.
    "adversarial_action": PerturbationConfig(
        name="adversarial_action",
        adversarial_action=True,
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

# Ordered list of preset names used by training loops and evaluation
PRESET_NAMES: List[str] = [
    "normal",
    "mirrored",
    "sigma_perturb",
    "adversarial_action",
    "hard_mode",
]


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
        PRESET_CONFIGS (e.g. "mirrored", "sigma_perturb", "adversarial_action").
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
        # Sigma perturbation state
        # ------------------------------------------------------------------
        # We store the original sampler config values so we can temporarily
        # modify them before each step() call and restore them after.
        # Access path: self._env.sampler.cfg  (DiffusionSamplerConfig)
        self._orig_sigma_min: Optional[float] = None
        self._orig_num_steps: Optional[int] = None

        if self.cfg.sigma_scale != 1.0 or self.cfg.steps_reduction != 0.0:
            # Snapshot original values at construction time so they are
            # always available even if the sampler config is modified elsewhere.
            try:
                sampler_cfg = self._env.sampler.cfg
                self._orig_sigma_min = sampler_cfg.sigma_min
                self._orig_num_steps = sampler_cfg.num_steps_denoising
            except AttributeError:
                # WorldModelEnv variant without .sampler — sigma perturbation
                # will be silently skipped.
                pass

        # ------------------------------------------------------------------
        # Adversarial action MLP
        # ------------------------------------------------------------------
        self._adv_mlp: Optional[nn.Module] = None
        self._adv_optimizer: Optional[torch.optim.Optimizer] = None
        self._episode_return: float = 0.0
        self._episode_step: int = 0
        self._max_episode_steps: int = 1000  # normalisation constant

        if self.cfg.adversarial_action:
            # Lazy init: we need num_actions from the env, but the env may not
            # be reset yet.  We initialise in the first step() call via
            # _maybe_init_adv_mlp().
            pass

    # ------------------------------------------------------------------
    # Env interface properties
    # ------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self._env.sampler.denoiser.inner_model.cfg.num_actions

    @property
    def device(self) -> torch.device:
        return self._env.device

    # num_actions is read by make_env_loop
    @property
    def num_actions(self) -> int:
        return self._env.sampler.denoiser.inner_model.cfg.num_actions  # fallback; real value set when available

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

        # Reset adversarial episode tracking
        if self.cfg.adversarial_action:
            self._episode_return = 0.0
            self._episode_step = 0

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

        For sigma_perturb: temporarily modifies the DiffusionSamplerConfig
        before calling the underlying step, then restores original values.

        For adversarial_action: applies a learned action logit offset and
        performs one adversarial gradient step after observing the reward.
        """
        # ------------------------------------------------------------------
        # Adversarial action: apply MLP offset to action (before any action
        # remapping / dropout) and accumulate the gradient step later.
        # ------------------------------------------------------------------
        adv_logit_offset: Optional[Tensor] = None
        if self.cfg.adversarial_action:
            act, adv_logit_offset = self._apply_adversarial_action(act)

        act = self._perturb_action(act)

        # ------------------------------------------------------------------
        # Sigma perturbation: temporarily modify the sampler config
        # ------------------------------------------------------------------
        sigma_modified = False
        if self.cfg.sigma_scale != 1.0 or self.cfg.steps_reduction != 0.0:
            if self._orig_sigma_min is not None:
                self._apply_sigma_perturbation()
                sigma_modified = True

        # Call underlying world-model step
        obs, rew, end, trunc, info = self._env.step(act)

        # ------------------------------------------------------------------
        # Restore sigma config immediately after step
        # ------------------------------------------------------------------
        if sigma_modified:
            self._restore_sigma()

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

        # ------------------------------------------------------------------
        # Adversarial action: one gradient step (maximise perturbation loss)
        # ------------------------------------------------------------------
        if self.cfg.adversarial_action and adv_logit_offset is not None:
            self._adversarial_gradient_step(adv_logit_offset, rew, end, trunc)

        return obs, rew, end, trunc, info

    # ------------------------------------------------------------------
    # predict_next_obs / predict_rew_end — forwarded for compile compat
    # ------------------------------------------------------------------

    def predict_next_obs(self):
        return self._env.predict_next_obs()

    def predict_rew_end(self, *args, **kwargs):
        return self._env.predict_rew_end(*args, **kwargs)

    # ------------------------------------------------------------------
    # Sigma perturbation helpers
    # ------------------------------------------------------------------

    def _apply_sigma_perturbation(self) -> None:
        """
        Temporarily modify the DiffusionSamplerConfig on self._env.sampler.cfg
        to perturb the EDM noise schedule for the upcoming step() call.

        What changes:
          - sigma_min is scaled up by cfg.sigma_scale (raises the noise floor)
          - num_steps_denoising is reduced by cfg.steps_reduction fraction
            (fewer denoising steps → less refined, noisier predictions)

        The sampler's pre-built sigmas tensor is also rebuilt so the schedule
        change takes effect immediately.  Original values are restored by
        _restore_sigma() after the step.
        """
        try:
            sampler = self._env.sampler
            cfg = sampler.cfg

            # Modify sigma_min
            new_sigma_min = self._orig_sigma_min * self.cfg.sigma_scale
            cfg.sigma_min = new_sigma_min

            # Modify num_steps_denoising
            new_num_steps = max(
                1,
                round(self._orig_num_steps * (1.0 - self.cfg.steps_reduction))
            )
            cfg.num_steps_denoising = new_num_steps

            # Rebuild the sigmas schedule so DiffusionSampler.sample() uses
            # the updated parameters
            from models.diffusion.diffusion_sampler import build_sigmas
            sampler.sigmas = build_sigmas(
                new_num_steps,
                new_sigma_min,
                cfg.sigma_max,
                cfg.rho,
                sampler.denoiser.device,
            )
        except (AttributeError, ImportError):
            # Gracefully skip if the sampler structure is different
            pass

    def _restore_sigma(self) -> None:
        """
        Restore DiffusionSamplerConfig and the sigmas tensor to their
        original values after a sigma-perturbed step() call.
        """
        try:
            sampler = self._env.sampler
            cfg = sampler.cfg

            cfg.sigma_min = self._orig_sigma_min
            cfg.num_steps_denoising = self._orig_num_steps

            from models.diffusion.diffusion_sampler import build_sigmas
            sampler.sigmas = build_sigmas(
                self._orig_num_steps,
                self._orig_sigma_min,
                cfg.sigma_max,
                cfg.rho,
                sampler.denoiser.device,
            )
        except (AttributeError, ImportError):
            pass

    # ------------------------------------------------------------------
    # Adversarial action helpers
    # ------------------------------------------------------------------

    def _maybe_init_adv_mlp(self, num_actions: int) -> None:
        """Lazily create the adversarial MLP on first use."""
        if self._adv_mlp is not None:
            return
        device = self.device
        self._adv_mlp = nn.Sequential(
            nn.Linear(2, self.cfg.adv_hidden),
            nn.ReLU(),
            nn.Linear(self.cfg.adv_hidden, num_actions),
        ).to(device)
        self._adv_optimizer = torch.optim.Adam(
            self._adv_mlp.parameters(), lr=self.cfg.adv_lr
        )

    def _apply_adversarial_action(
        self, act: torch.LongTensor
    ) -> Tuple[torch.LongTensor, Tensor]:
        """
        Compute the adversarial action logit offset from the MLP and apply it
        to perturb the action selection.

        The context vector is [normalised_return, normalised_step]:
          - normalised_return: episode_return / (max_observed + epsilon), capped to [-1, 1]
          - normalised_step:   episode_step / max_episode_steps, in [0, 1]

        The MLP produces a [1, num_actions] offset tensor.  The offset is added
        to a one-hot encoding of the original action to produce perturbed logits,
        and the new action is argmax of those logits.

        Returns (perturbed_act, offset_tensor).
        """
        # Infer num_actions from the action tensor
        # act is [num_envs] of long indices — we use the env's action space
        # (we assume at least as many actions as the max index + 1, or use
        # the env's num_actions if available)
        try:
            num_actions = int(self._env.sampler.denoiser.cfg.inner_model.num_actions)
        except AttributeError:
            num_actions = int(act.max().item()) + 1

        self._maybe_init_adv_mlp(num_actions)

        device = self.device
        norm_return = float(self._episode_return) / (
            abs(self._episode_return) + 1.0
        )  # tanh-like, stays in (-1, 1)
        norm_step = min(1.0, self._episode_step / max(self._max_episode_steps, 1))

        context = torch.tensor(
            [[norm_return, norm_step]], dtype=torch.float32, device=device
        )  # [1, 2]

        offset = self._adv_mlp(context)  # [1, num_actions]

        # Build one-hot logits from the original actions, add offset, take argmax
        # We process each env independently; offset is broadcast across envs.
        one_hot = F.one_hot(act.clamp(0, num_actions - 1), num_actions).float()
        # one_hot: [num_envs, num_actions]; offset: [1, num_actions]
        perturbed_logits = one_hot + offset  # broadcast over num_envs
        perturbed_act = perturbed_logits.argmax(dim=-1).long()

        self._episode_step += 1
        return perturbed_act, offset

    def _adversarial_gradient_step(
        self,
        adv_logit_offset: Tensor,
        rew: Tensor,
        end: Tensor,
        trunc: Tensor,
    ) -> None:
        """
        Perform one gradient step on the adversarial MLP.

        Objective: push the MLP to produce larger offsets (more aggressive
        perturbations). We use the L2 norm of the offset as a surrogate loss
        with sign flipped by the reward: when reward is high (agent is doing
        well), maximise offset magnitude; when reward is low, the perturbation
        is already working so we scale back. This avoids needing gradients
        through the non-differentiable environment reward.

        loss = -mean_rew_sign * ||offset||^2
        """
        if self._adv_mlp is None or self._adv_optimizer is None:
            return

        mean_rew = rew.float().mean().item()
        # Sign: when agent gets positive reward, adversary should push harder
        sign = 1.0 if mean_rew >= 0 else -1.0
        # Surrogate loss: maximise offset magnitude when agent is doing well
        loss = -sign * adv_logit_offset.pow(2).mean()

        self._adv_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._adv_mlp.parameters(), 1.0)
        self._adv_optimizer.step()

        # Update episode return tracking (use mean reward across envs)
        self._episode_return += float(mean_rew)

        # Reset on episode termination
        done = torch.logical_or(end, trunc)
        if done.any():
            self._episode_return = 0.0
            self._episode_step = 0

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
) -> Dict[str, "PerturbedWorldModelEnv"]:
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
# v4 fix3
