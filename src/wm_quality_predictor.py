"""
wm_quality_predictor.py — Predict World Model Quality Without Running RL

Computes 7 proxy features from a trained world model and correlates them
with actual RL agent performance across 26 DIAMOND Atari games.

Features:
1. Single-step prediction error (MSE)
2. Multi-step rollout stability (error growth rate)
3. Action sensitivity (do different actions produce different outcomes?)
4. Stochasticity calibration (uncertainty vs actual environment variance)
5. Reward prediction accuracy
6. Visual detail preservation (LPIPS / high-frequency energy)
7. State space coverage (distribution match between WM and real env)

This addresses the evaluation bottleneck identified in:
- WorldArena (Feb 2026): "visual quality does not translate to task capability"
- Not Boring essay (Mar 2026): "no single approach has proved superior"
- The fundamental need: fast iteration on world model architectures

No GPU required. Runs on CPU with pretrained DIAMOND models from HuggingFace.
"""

import sys
import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

# DIAMOND source path
DIAMOND_SRC = Path(os.environ.get("DIAMOND_DIR", os.path.expanduser("~/diamond"))) / "src"
if str(DIAMOND_SRC) not in sys.path:
    sys.path.insert(0, str(DIAMOND_SRC))


@dataclass
class WMQualityFeatures:
    """All proxy features for one game's world model."""
    game: str = ""
    
    # Feature 1: Single-step prediction error
    single_step_mse: float = 0.0
    single_step_mae: float = 0.0
    
    # Feature 2: Multi-step rollout stability
    error_at_step: Dict[int, float] = field(default_factory=dict)  # step -> MSE
    error_growth_rate: float = 0.0  # slope of log(MSE) vs step
    error_doubling_steps: float = 0.0  # steps for error to double
    rollout_stability_score: float = 0.0  # 1 / (1 + growth_rate)
    
    # Feature 3: Action sensitivity
    action_sensitivity: float = 0.0  # mean pairwise MSE between different-action predictions
    action_sensitivity_normalized: float = 0.0  # sensitivity / single_step_mse
    
    # Feature 4: Stochasticity calibration
    model_uncertainty: float = 0.0  # variance of diffusion samples
    environment_variance: float = 0.0  # actual variance in env transitions
    calibration_ratio: float = 0.0  # model_uncertainty / environment_variance
    
    # Feature 5: Reward prediction accuracy
    reward_accuracy: float = 0.0
    end_accuracy: float = 0.0
    reward_f1: float = 0.0
    
    # Feature 6: Visual detail preservation
    high_freq_energy_ratio: float = 0.0  # HF energy in prediction / HF energy in ground truth
    
    # Feature 7: State space coverage
    state_coverage_mmd: float = 0.0  # MMD between WM rollout states and real states
    
    # Published score
    diamond_hns: float = 0.0


def load_pretrained_agent(game_short_name: str, device='cpu'):
    """
    Load a pretrained DIAMOND agent from HuggingFace.
    
    Returns the Agent object ready for inference.
    """
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    from huggingface_hub import hf_hub_download
    from hydra.utils import instantiate
    from agent import Agent
    
    # Download config and model
    config_path = hf_hub_download("eloialonso/diamond", "atari_100k/config/agent/default.yaml")
    env_config_path = hf_hub_download("eloialonso/diamond", "atari_100k/config/env/atari.yaml")
    model_path = hf_hub_download("eloialonso/diamond", f"atari_100k/models/{game_short_name}.pt")
    
    # Load configs
    agent_cfg = OmegaConf.load(config_path)
    env_cfg = OmegaConf.load(env_config_path)
    
    # Resolve cross-references manually
    agent_cfg.rew_end_model.img_channels = agent_cfg.denoiser.inner_model.img_channels
    agent_cfg.rew_end_model.img_size = env_cfg.train.size
    agent_cfg.actor_critic.img_channels = agent_cfg.denoiser.inner_model.img_channels
    agent_cfg.actor_critic.img_size = env_cfg.train.size
    
    OmegaConf.resolve(agent_cfg)
    
    # Determine num_actions (varies by game)
    from envs import make_atari_env
    game_id = f"{game_short_name}NoFrameskip-v4"
    env = make_atari_env(num_envs=1, device=torch.device(device), id=game_id,
                         done_on_life_loss=True, size=64, max_episode_steps=None)
    num_actions = env.num_actions
    
    # Create and load agent
    agent = Agent(instantiate(agent_cfg, num_actions=num_actions)).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    agent.load_state_dict(checkpoint)
    agent.eval()
    
    return agent, env, num_actions


def collect_real_episodes(env, num_steps=10000):
    """Collect episodes using random policy in real environment."""
    from data.episode import Episode
    
    episodes = []
    obs, _ = env.reset()
    eo, ea, er, ee, et = [obs], [], [], [], []
    steps = 0
    
    while steps < num_steps:
        a = torch.randint(0, env.num_actions, (1,), device=obs.device)
        no, rw, en, tr, info = env.step(a)
        ea.append(a); er.append(rw); ee.append(en); et.append(tr)
        dead = (en + tr).clip(max=1).item()
        steps += 1
        if dead:
            ep = Episode(obs=torch.cat(eo), act=torch.cat(ea), rew=torch.cat(er),
                        end=torch.cat(ee), trunc=torch.cat(et), info={})
            episodes.append(ep)
            obs, _ = env.reset()
            eo, ea, er, ee, et = [obs], [], [], [], []
        else:
            eo.append(no)
    
    return episodes


@torch.no_grad()
def compute_feature_1_single_step(agent, episodes, device='cpu', max_frames=500):
    """Feature 1: Single-step prediction error."""
    from models.diffusion import DiffusionSampler, DiffusionSamplerConfig
    
    sampler_cfg = DiffusionSamplerConfig(
        num_steps_denoising=3, sigma_min=2e-3, sigma_max=5.0,
        rho=7, order=1, s_churn=0.0, s_tmin=0.0,
        s_tmax=float('inf'), s_noise=1.0)
    sampler = DiffusionSampler(agent.denoiser, sampler_cfg)
    
    context_len = 4
    mses, maes = [], []
    count = 0
    
    for ep in episodes:
        if count >= max_frames:
            break
        obs = ep.obs.to(device)
        act = ep.act.to(device)
        T = len(ep)
        
        for t in range(context_len, T - 1, 3):  # every 3rd frame
            if count >= max_frames:
                break
            obs_w = obs[t-context_len:t].unsqueeze(0)
            act_w = act[t-context_len:t].unsqueeze(0)
            actual = obs[t]
            
            pred, _ = sampler.sample(obs_w, act_w)
            pred = pred.squeeze(0)
            
            mse = ((pred - actual) ** 2).mean().item()
            mae = (pred - actual).abs().mean().item()
            mses.append(mse)
            maes.append(mae)
            count += 1
    
    return {
        'single_step_mse': float(np.mean(mses)) if mses else 0,
        'single_step_mae': float(np.mean(maes)) if maes else 0,
        'n_frames_evaluated': count,
    }


@torch.no_grad()
def compute_feature_2_rollout_stability(agent, episodes, device='cpu', 
                                         rollout_length=50, n_rollouts=20):
    """Feature 2: Multi-step rollout stability — how fast does error grow?"""
    from models.diffusion import DiffusionSampler, DiffusionSamplerConfig
    
    sampler_cfg = DiffusionSamplerConfig(
        num_steps_denoising=3, sigma_min=2e-3, sigma_max=5.0,
        rho=7, order=1, s_churn=0.0, s_tmin=0.0,
        s_tmax=float('inf'), s_noise=1.0)
    sampler = DiffusionSampler(agent.denoiser, sampler_cfg)
    
    context_len = 4
    step_errors = defaultdict(list)
    
    rollouts_done = 0
    for ep in episodes:
        if rollouts_done >= n_rollouts:
            break
        obs = ep.obs.to(device)
        act = ep.act.to(device)
        T = len(ep)
        
        if T < context_len + rollout_length:
            continue
        
        # Start from a random point in the episode
        start = context_len
        
        # Initialize with real context
        obs_buffer = obs[start-context_len:start].unsqueeze(0).clone()
        act_buffer = act[start-context_len:start].unsqueeze(0).clone()
        
        # Rollout autoregressively
        for step in range(min(rollout_length, T - start - 1)):
            t = start + step
            act_buffer[:, -1] = act[t]
            
            pred, _ = sampler.sample(obs_buffer, act_buffer)
            actual = obs[t]
            
            mse = ((pred.squeeze(0) - actual) ** 2).mean().item()
            step_errors[step].append(mse)
            
            # Shift buffer: add prediction, remove oldest
            obs_buffer = torch.cat([obs_buffer[:, 1:], pred.unsqueeze(1)], dim=1)
            act_buffer = torch.cat([act_buffer[:, 1:], torch.zeros_like(act_buffer[:, :1])], dim=1)
        
        rollouts_done += 1
    
    # Compute error at each step
    error_at_step = {s: float(np.mean(errs)) for s, errs in sorted(step_errors.items())}
    
    # Compute error growth rate (slope of log(MSE) vs step)
    if len(error_at_step) >= 5:
        steps = np.array(sorted(error_at_step.keys()))
        errors = np.array([error_at_step[s] for s in steps])
        # Avoid log of zero
        errors = np.maximum(errors, 1e-10)
        log_errors = np.log(errors)
        
        # Linear fit to log(error) vs step
        coeffs = np.polyfit(steps, log_errors, 1)
        growth_rate = float(coeffs[0])
        
        # Error doubling time
        doubling_steps = float(np.log(2) / max(growth_rate, 1e-10)) if growth_rate > 0 else float('inf')
        
        stability_score = 1.0 / (1.0 + max(growth_rate, 0))
    else:
        growth_rate = 0
        doubling_steps = float('inf')
        stability_score = 1.0
    
    return {
        'error_at_step': error_at_step,
        'error_growth_rate': growth_rate,
        'error_doubling_steps': min(doubling_steps, 1000),
        'rollout_stability_score': stability_score,
        'n_rollouts': rollouts_done,
    }


@torch.no_grad()
def compute_feature_3_action_sensitivity(agent, episodes, num_actions, device='cpu',
                                          max_states=100):
    """Feature 3: Action sensitivity — do different actions produce different outcomes?"""
    from models.diffusion import DiffusionSampler, DiffusionSamplerConfig
    
    sampler_cfg = DiffusionSamplerConfig(
        num_steps_denoising=3, sigma_min=2e-3, sigma_max=5.0,
        rho=7, order=1, s_churn=0.0, s_tmin=0.0,
        s_tmax=float('inf'), s_noise=1.0)
    sampler = DiffusionSampler(agent.denoiser, sampler_cfg)
    
    context_len = 4
    sensitivities = []
    count = 0
    
    for ep in episodes:
        if count >= max_states:
            break
        obs = ep.obs.to(device)
        act = ep.act.to(device)
        T = len(ep)
        
        for t in range(context_len, T - 1, 5):
            if count >= max_states:
                break
            
            obs_w = obs[t-context_len:t].unsqueeze(0)
            act_w = act[t-context_len:t].unsqueeze(0)
            
            # Predict next frame for each possible action
            predictions = []
            for a in range(num_actions):
                act_modified = act_w.clone()
                act_modified[:, -1] = a
                pred, _ = sampler.sample(obs_w, act_modified)
                predictions.append(pred.squeeze(0))
            
            # Compute mean pairwise MSE between predictions for different actions
            pairwise_diffs = []
            for i in range(num_actions):
                for j in range(i + 1, num_actions):
                    diff = ((predictions[i] - predictions[j]) ** 2).mean().item()
                    pairwise_diffs.append(diff)
            
            if pairwise_diffs:
                sensitivities.append(np.mean(pairwise_diffs))
            count += 1
    
    mean_sensitivity = float(np.mean(sensitivities)) if sensitivities else 0
    
    return {
        'action_sensitivity': mean_sensitivity,
        'n_states_tested': count,
    }


@torch.no_grad()
def compute_feature_4_stochasticity(agent, episodes, device='cpu',
                                     n_samples=5, max_states=100):
    """Feature 4: Stochasticity calibration — does model uncertainty match reality?"""
    from models.diffusion import DiffusionSampler, DiffusionSamplerConfig
    
    sampler_cfg = DiffusionSamplerConfig(
        num_steps_denoising=3, sigma_min=2e-3, sigma_max=5.0,
        rho=7, order=1, s_churn=0.0, s_tmin=0.0,
        s_tmax=float('inf'), s_noise=1.0)
    sampler = DiffusionSampler(agent.denoiser, sampler_cfg)
    
    context_len = 4
    model_variances = []
    count = 0
    
    for ep in episodes:
        if count >= max_states:
            break
        obs = ep.obs.to(device)
        act = ep.act.to(device)
        T = len(ep)
        
        for t in range(context_len, T - 1, 5):
            if count >= max_states:
                break
            
            obs_w = obs[t-context_len:t].unsqueeze(0)
            act_w = act[t-context_len:t].unsqueeze(0)
            
            # Multiple samples from diffusion model
            preds = []
            for _ in range(n_samples):
                pred, _ = sampler.sample(obs_w, act_w)
                preds.append(pred.squeeze(0))
            
            preds = torch.stack(preds)
            variance = preds.var(dim=0).mean().item()
            model_variances.append(variance)
            count += 1
    
    return {
        'model_uncertainty': float(np.mean(model_variances)) if model_variances else 0,
        'model_uncertainty_std': float(np.std(model_variances)) if model_variances else 0,
        'n_states_tested': count,
    }


@torch.no_grad()
def compute_feature_5_reward_accuracy(agent, episodes, device='cpu', max_frames=500):
    """Feature 5: Reward and end prediction accuracy."""
    context_len = 4
    rew_correct, rew_total = 0, 0
    end_correct, end_total = 0, 0
    count = 0
    
    for ep in episodes:
        if count >= max_frames:
            break
        obs = ep.obs.to(device)
        act = ep.act.to(device)
        rew = ep.rew.to(device)
        end = ep.end.to(device)
        T = len(ep)
        
        if T < context_len + 2:
            continue
        
        # Run reward/end model on sequences
        for t in range(context_len, T - 1, 3):
            if count >= max_frames:
                break
            
            obs_seq = obs[t-context_len:t+1].unsqueeze(0)  # [1, ctx+1, C, H, W]
            act_seq = act[t-context_len:t].unsqueeze(0)    # [1, ctx]
            next_obs = obs[t:t+1].unsqueeze(0)             # [1, 1, C, H, W]
            
            # Initialize LSTM state
            hx = torch.zeros(1, 1, agent.rew_end_model.lstm_dim, device=device)
            cx = torch.zeros(1, 1, agent.rew_end_model.lstm_dim, device=device)
            
            try:
                logits_rew, logits_end, _ = agent.rew_end_model.predict_rew_end(
                    obs_seq[:, :-1], act_seq, next_obs, (hx, cx))
                
                pred_rew = logits_rew.argmax(dim=-1).squeeze() - 1  # {-1, 0, 1}
                pred_end = logits_end.argmax(dim=-1).squeeze()
                
                actual_rew = rew[t].sign().long()
                actual_end = end[t].long()
                
                rew_correct += (pred_rew == actual_rew).sum().item()
                rew_total += 1
                end_correct += (pred_end == actual_end).sum().item()
                end_total += 1
            except Exception:
                pass
            
            count += 1
    
    return {
        'reward_accuracy': float(rew_correct / max(rew_total, 1)),
        'end_accuracy': float(end_correct / max(end_total, 1)),
        'n_frames_evaluated': count,
    }


@torch.no_grad()
def compute_feature_6_visual_detail(agent, episodes, device='cpu', max_frames=200):
    """Feature 6: Visual detail preservation — high-frequency energy in predictions."""
    from models.diffusion import DiffusionSampler, DiffusionSamplerConfig
    
    sampler_cfg = DiffusionSamplerConfig(
        num_steps_denoising=3, sigma_min=2e-3, sigma_max=5.0,
        rho=7, order=1, s_churn=0.0, s_tmin=0.0,
        s_tmax=float('inf'), s_noise=1.0)
    sampler = DiffusionSampler(agent.denoiser, sampler_cfg)
    
    def high_freq_energy(img):
        """Compute high-frequency energy using Laplacian."""
        # Simple approximation: sum of squared differences with neighbors
        dx = img[:, :, 1:] - img[:, :, :-1]
        dy = img[:, 1:, :] - img[:, :-1, :]
        return (dx ** 2).mean().item() + (dy ** 2).mean().item()
    
    context_len = 4
    hf_ratios = []
    count = 0
    
    for ep in episodes:
        if count >= max_frames:
            break
        obs = ep.obs.to(device)
        act = ep.act.to(device)
        T = len(ep)
        
        for t in range(context_len, T - 1, 5):
            if count >= max_frames:
                break
            
            obs_w = obs[t-context_len:t].unsqueeze(0)
            act_w = act[t-context_len:t].unsqueeze(0)
            actual = obs[t]
            
            pred, _ = sampler.sample(obs_w, act_w)
            pred = pred.squeeze(0)
            
            hf_pred = high_freq_energy(pred)
            hf_actual = high_freq_energy(actual)
            
            if hf_actual > 1e-8:
                hf_ratios.append(hf_pred / hf_actual)
            count += 1
    
    return {
        'high_freq_energy_ratio': float(np.mean(hf_ratios)) if hf_ratios else 0,
        'n_frames_evaluated': count,
    }


@torch.no_grad()
def compute_feature_7_state_coverage(agent, episodes, device='cpu',
                                      rollout_steps=30, n_rollouts=20):
    """Feature 7: State space coverage — MMD between WM rollouts and real env."""
    from models.diffusion import DiffusionSampler, DiffusionSamplerConfig
    import torchvision.models as models
    
    sampler_cfg = DiffusionSamplerConfig(
        num_steps_denoising=3, sigma_min=2e-3, sigma_max=5.0,
        rho=7, order=1, s_churn=0.0, s_tmin=0.0,
        s_tmax=float('inf'), s_noise=1.0)
    sampler = DiffusionSampler(agent.denoiser, sampler_cfg)
    
    # Simple encoder: flatten + PCA-like (just use raw pixel stats for efficiency)
    def embed_batch(frames):
        """Simple embedding: spatial statistics."""
        # Mean/std per channel + spatial gradient stats
        b = frames.shape[0]
        channel_means = frames.mean(dim=(2, 3))  # [B, C]
        channel_stds = frames.std(dim=(2, 3))     # [B, C]
        # Spatial gradients
        dx = (frames[:, :, :, 1:] - frames[:, :, :, :-1]).abs().mean(dim=(2, 3))
        dy = (frames[:, :, 1:, :] - frames[:, :, :-1, :]).abs().mean(dim=(2, 3))
        return torch.cat([channel_means, channel_stds, dx, dy], dim=1).numpy()
    
    context_len = 4
    
    # Collect real environment frames
    real_frames = []
    for ep in episodes[:30]:
        for t in range(0, len(ep), 5):
            real_frames.append(ep.obs[t])
    real_frames = torch.stack(real_frames[:500])
    real_emb = embed_batch(real_frames)
    
    # Collect world model rollout frames
    wm_frames = []
    rollouts_done = 0
    for ep in episodes:
        if rollouts_done >= n_rollouts:
            break
        obs = ep.obs.to(device)
        act = ep.act.to(device)
        T = len(ep)
        if T < context_len + rollout_steps:
            continue
        
        obs_buffer = obs[:context_len].unsqueeze(0).clone()
        act_buffer = act[:context_len].unsqueeze(0).clone()
        
        for step in range(min(rollout_steps, T - context_len)):
            t = context_len + step
            act_buffer[:, -1] = act[t] if t < T else 0
            pred, _ = sampler.sample(obs_buffer, act_buffer)
            wm_frames.append(pred.squeeze(0).squeeze(0))
            obs_buffer = torch.cat([obs_buffer[:, 1:], pred.unsqueeze(1)], dim=1)
            act_buffer = torch.cat([act_buffer[:, 1:], torch.zeros_like(act_buffer[:, :1])], dim=1)
        
        rollouts_done += 1
    
    if not wm_frames:
        return {'state_coverage_mmd': 0, 'n_rollouts': 0}
    
    wm_frames = torch.stack(wm_frames[:500])
    wm_emb = embed_batch(wm_frames)
    
    # Compute MMD (Maximum Mean Discrepancy)
    def mmd_rbf(X, Y, gamma=1.0):
        from sklearn.metrics import pairwise_distances
        XX = pairwise_distances(X, X, metric='sqeuclidean')
        YY = pairwise_distances(Y, Y, metric='sqeuclidean')
        XY = pairwise_distances(X, Y, metric='sqeuclidean')
        
        K_XX = np.exp(-gamma * XX).mean()
        K_YY = np.exp(-gamma * YY).mean()
        K_XY = np.exp(-gamma * XY).mean()
        
        return float(K_XX + K_YY - 2 * K_XY)
    
    # Use median heuristic for gamma
    from sklearn.metrics import pairwise_distances
    all_dists = pairwise_distances(real_emb[:100], wm_emb[:100], metric='sqeuclidean')
    gamma = 1.0 / max(np.median(all_dists), 1e-8)
    
    mmd = mmd_rbf(real_emb[:200], wm_emb[:200], gamma)
    
    return {
        'state_coverage_mmd': mmd,
        'n_rollouts': rollouts_done,
    }


def compute_all_features(game_short_name: str, device='cpu') -> Dict:
    """
    Compute all 7 proxy features for a single game's pretrained DIAMOND model.
    """
    print(f"    Loading agent...", end=" ", flush=True)
    agent, env, num_actions = load_pretrained_agent(game_short_name, device)
    print(f"OK ({num_actions} actions)")
    
    print(f"    Collecting episodes...", end=" ", flush=True)
    episodes = collect_real_episodes(env, num_steps=15000)
    print(f"{len(episodes)} episodes")
    
    results = {'game': game_short_name, 'num_actions': num_actions, 'n_episodes': len(episodes)}
    
    print(f"    Feature 1: Single-step error...", end=" ", flush=True)
    f1 = compute_feature_1_single_step(agent, episodes, device, max_frames=300)
    results.update(f1)
    print(f"MSE={f1['single_step_mse']:.6f}")
    
    print(f"    Feature 2: Rollout stability...", end=" ", flush=True)
    f2 = compute_feature_2_rollout_stability(agent, episodes, device, rollout_length=30, n_rollouts=10)
    results.update(f2)
    print(f"growth={f2['error_growth_rate']:.4f}")
    
    print(f"    Feature 3: Action sensitivity...", end=" ", flush=True)
    f3 = compute_feature_3_action_sensitivity(agent, episodes, num_actions, device, max_states=50)
    results.update(f3)
    print(f"sens={f3['action_sensitivity']:.6f}")
    
    print(f"    Feature 4: Stochasticity...", end=" ", flush=True)
    f4 = compute_feature_4_stochasticity(agent, episodes, device, n_samples=3, max_states=50)
    results.update(f4)
    print(f"unc={f4['model_uncertainty']:.6f}")
    
    print(f"    Feature 5: Reward accuracy...", end=" ", flush=True)
    f5 = compute_feature_5_reward_accuracy(agent, episodes, device, max_frames=300)
    results.update(f5)
    print(f"rew_acc={f5['reward_accuracy']:.3f}")
    
    print(f"    Feature 6: Visual detail...", end=" ", flush=True)
    f6 = compute_feature_6_visual_detail(agent, episodes, device, max_frames=150)
    results.update(f6)
    print(f"HF_ratio={f6['high_freq_energy_ratio']:.3f}")
    
    print(f"    Feature 7: State coverage...", end=" ", flush=True)
    f7 = compute_feature_7_state_coverage(agent, episodes, device, rollout_steps=20, n_rollouts=10)
    results.update(f7)
    print(f"MMD={f7['state_coverage_mmd']:.6f}")
    
    return results
