"""
failure_eval.py — Failure-Conditioned Evaluation Metrics for World Models

Standard world model evaluation measures average visual fidelity (FID, SSIM, FVD)
or average agent return. But these metrics miss the critical question:

    "How well does the world model handle failure/edge-case moments?"

This module implements four failure-conditioned metrics:

1. Failure Prediction Error (FPE): MSE ratio between failure-frame and
   normal-frame predictions. FPE > 1.0 means the model is worse at failures.

2. Edge Case Uncertainty Calibration: Does the model's diffusion sampling
   variance correlate with actual prediction difficulty? A well-calibrated
   model should be MORE uncertain at genuinely unpredictable moments.

3. Recovery Prediction Accuracy: After a failure event (life loss), how well
   does the world model predict the post-death state transition?

4. Failure Boundary Sharpness: How prediction error changes as frames
   approach a failure event. Sharp boundary = model can't anticipate failures.
   Gradual curve = model is learning failure precursors.

These metrics address the evaluation gap identified in:
- WorldArena (Feb 2026): "high visual quality does not translate into strong task capability"
- 1X World Model: "failure data from autonomous rollouts is critical"
- Not Boring essay (Mar 2026): Pim de Witte calling for understanding "how much data of which type"

Reference: arxiv.org/abs/2602.08971 (WorldArena), 1x.tech/1x-world-model.pdf
"""

import sys
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Ensure DIAMOND's src is importable
DIAMOND_SRC = Path.home() / "diamond" / "src"
if str(DIAMOND_SRC) not in sys.path:
    sys.path.insert(0, str(DIAMOND_SRC))


@dataclass
class FailureEvalResults:
    """Results from failure-conditioned evaluation."""
    # Core metrics
    failure_prediction_error: float = 0.0   # Mean MSE on failure frames
    normal_prediction_error: float = 0.0    # Mean MSE on normal frames
    fpe_ratio: float = 0.0                  # failure / normal (>1 = worse at failures)
    
    # Uncertainty calibration
    failure_uncertainty: float = 0.0        # Mean diffusion variance on failure frames
    normal_uncertainty: float = 0.0         # Mean diffusion variance on normal frames
    uncertainty_ratio: float = 0.0          # failure / normal (>1 = more uncertain at failures)
    
    # Recovery accuracy
    recovery_prediction_error: float = 0.0  # MSE on frames just after life loss
    
    # Boundary sharpness
    boundary_errors: Dict[int, float] = field(default_factory=dict)  # distance_to_failure -> MSE
    boundary_sharpness: float = 0.0         # slope of error curve approaching failure
    
    # Counts
    n_failure_frames: int = 0
    n_normal_frames: int = 0
    n_recovery_frames: int = 0
    
    # Per-episode breakdowns
    per_episode_fpe: List[float] = field(default_factory=list)


class FailureConditionedEvaluator:
    """
    Evaluates a trained DIAMOND world model with failure-conditioned metrics.
    
    Usage:
        evaluator = FailureConditionedEvaluator(agent, device)
        results = evaluator.evaluate_on_episodes(episodes, failure_events)
    """
    
    def __init__(self, agent, device='cuda', n_diffusion_samples=5):
        """
        Args:
            agent: DIAMOND Agent object (contains denoiser, rew_end_model, actor_critic)
            device: torch device
            n_diffusion_samples: number of diffusion samples for uncertainty estimation
        """
        self.agent = agent
        self.device = device
        self.n_diffusion_samples = n_diffusion_samples
        self.agent.eval()
    
    @torch.no_grad()
    def predict_next_frame(self, obs_history, action):
        """
        Given observation history and action, predict next frame using the world model.
        
        Args:
            obs_history: [B, context_len, C, H, W] tensor
            action: [B, context_len] tensor of actions
            
        Returns:
            predictions: list of [B, C, H, W] tensors (multiple diffusion samples)
            mean_prediction: [B, C, H, W]
            uncertainty: [B, C, H, W] per-pixel std across samples
        """
        from models.diffusion import DiffusionSampler, DiffusionSamplerConfig
        
        sampler_cfg = DiffusionSamplerConfig(
            num_steps_denoising=3,
            sigma_min=2e-3,
            sigma_max=5.0,
            rho=7,
            order=1,
            s_churn=0.0,
            s_tmin=0.0,
            s_tmax=float('inf'),
            s_noise=1.0,
        )
        sampler = DiffusionSampler(self.agent.denoiser, sampler_cfg)
        
        predictions = []
        for _ in range(self.n_diffusion_samples):
            pred, _ = sampler.sample(obs_history, action)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [n_samples, B, C, H, W]
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return predictions, mean_pred, uncertainty
    
    def compute_frame_error(self, predicted, actual):
        """MSE between predicted and actual frame."""
        return ((predicted - actual) ** 2).mean(dim=(-3, -2, -1))  # per-batch-element MSE
    
    def evaluate_episode(
        self,
        episode,
        failure_frame_set: set,
        death_frames: set,
        context_length: int = 4,
    ) -> Dict:
        """
        Evaluate world model prediction quality on a single episode,
        splitting results by failure vs normal frames.
        
        Args:
            episode: DIAMOND Episode object
            failure_frame_set: set of frame indices that are failure-related
            death_frames: set of frame indices where end=1 (actual life loss)
            context_length: number of frames to condition on (DIAMOND uses 4)
        
        Returns:
            Dict with per-frame errors and uncertainties
        """
        obs = episode.obs.to(self.device)  # [T, C, H, W]
        act = episode.act.to(self.device)  # [T]
        T = len(episode)
        
        failure_errors = []
        normal_errors = []
        failure_uncertainties = []
        normal_uncertainties = []
        recovery_errors = []
        boundary_data = []  # (distance_to_nearest_death, error)
        
        # Find nearest death frame for each timestep
        death_list = sorted(death_frames) if death_frames else []
        
        for t in range(context_length, T - 1):
            # Prepare input: [1, context_len, C, H, W]
            obs_window = obs[t - context_length:t].unsqueeze(0)
            act_window = act[t - context_length:t].unsqueeze(0)
            actual_next = obs[t]
            
            # Predict
            _, mean_pred, uncertainty = self.predict_next_frame(obs_window, act_window)
            
            # Compute error
            error = self.compute_frame_error(mean_pred.squeeze(0), actual_next).item()
            unc = uncertainty.mean().item()
            
            # Classify frame
            is_failure = t in failure_frame_set
            is_recovery = any(d < t <= d + 3 for d in death_list)  # 3 frames after death
            
            if is_failure:
                failure_errors.append(error)
                failure_uncertainties.append(unc)
            else:
                normal_errors.append(error)
                normal_uncertainties.append(unc)
            
            if is_recovery:
                recovery_errors.append(error)
            
            # Distance to nearest death
            if death_list:
                distances = [d - t for d in death_list if d >= t]
                if distances:
                    dist = min(distances)
                    boundary_data.append((dist, error))
        
        return {
            'failure_errors': failure_errors,
            'normal_errors': normal_errors,
            'failure_uncertainties': failure_uncertainties,
            'normal_uncertainties': normal_uncertainties,
            'recovery_errors': recovery_errors,
            'boundary_data': boundary_data,
        }
    
    def evaluate_on_episodes(
        self,
        episodes: list,
        failure_events_per_episode: dict,
        max_episodes: int = 50,
        context_length: int = 4,
    ) -> FailureEvalResults:
        """
        Run failure-conditioned evaluation across multiple episodes.
        
        Args:
            episodes: list of DIAMOND Episode objects
            failure_events_per_episode: dict mapping episode_id -> list of FailureEvent
            max_episodes: cap on episodes to evaluate (for time)
            context_length: DIAMOND conditioning window
            
        Returns:
            FailureEvalResults with all metrics computed
        """
        all_failure_errors = []
        all_normal_errors = []
        all_failure_unc = []
        all_normal_unc = []
        all_recovery_errors = []
        all_boundary_data = []
        per_episode_fpe = []
        
        from failure_detector import FailureDetector
        detector = FailureDetector()
        
        n_eval = min(len(episodes), max_episodes)
        
        for ep_idx in range(n_eval):
            episode = episodes[ep_idx]
            
            # Get failure events for this episode
            events = failure_events_per_episode.get(ep_idx, [])
            if not events:
                events = detector.detect_from_episode(episode, ep_idx)
            
            failure_frame_set = detector.get_failure_frame_set(events)
            death_frames = {e.frame_idx for e in events if e.event_type == "life_loss"}
            
            if len(episode) <= context_length + 1:
                continue
            
            ep_results = self.evaluate_episode(
                episode, failure_frame_set, death_frames, context_length
            )
            
            all_failure_errors.extend(ep_results['failure_errors'])
            all_normal_errors.extend(ep_results['normal_errors'])
            all_failure_unc.extend(ep_results['failure_uncertainties'])
            all_normal_unc.extend(ep_results['normal_uncertainties'])
            all_recovery_errors.extend(ep_results['recovery_errors'])
            all_boundary_data.extend(ep_results['boundary_data'])
            
            # Per-episode FPE
            if ep_results['failure_errors'] and ep_results['normal_errors']:
                ep_fpe = np.mean(ep_results['failure_errors']) / np.mean(ep_results['normal_errors'])
                per_episode_fpe.append(ep_fpe)
            
            if (ep_idx + 1) % 10 == 0:
                print(f"  Evaluated {ep_idx + 1}/{n_eval} episodes")
        
        # Compute aggregate metrics
        results = FailureEvalResults()
        
        if all_failure_errors:
            results.failure_prediction_error = np.mean(all_failure_errors)
            results.n_failure_frames = len(all_failure_errors)
        if all_normal_errors:
            results.normal_prediction_error = np.mean(all_normal_errors)
            results.n_normal_frames = len(all_normal_errors)
        if all_failure_errors and all_normal_errors:
            results.fpe_ratio = results.failure_prediction_error / results.normal_prediction_error
        
        if all_failure_unc:
            results.failure_uncertainty = np.mean(all_failure_unc)
        if all_normal_unc:
            results.normal_uncertainty = np.mean(all_normal_unc)
        if all_failure_unc and all_normal_unc:
            results.uncertainty_ratio = results.failure_uncertainty / results.normal_uncertainty
        
        if all_recovery_errors:
            results.recovery_prediction_error = np.mean(all_recovery_errors)
            results.n_recovery_frames = len(all_recovery_errors)
        
        # Boundary sharpness: bin by distance to failure
        if all_boundary_data:
            boundary_bins = defaultdict(list)
            for dist, err in all_boundary_data:
                if dist <= 20:  # Only look at frames within 20 steps of death
                    boundary_bins[dist].append(err)
            
            results.boundary_errors = {d: np.mean(errs) for d, errs in sorted(boundary_bins.items())}
            
            # Sharpness = slope of error from dist=10 to dist=0
            if len(results.boundary_errors) >= 3:
                dists = sorted(results.boundary_errors.keys())
                errors_at_dist = [results.boundary_errors[d] for d in dists]
                if len(dists) >= 2:
                    # Linear regression slope
                    coeffs = np.polyfit(dists, errors_at_dist, 1)
                    results.boundary_sharpness = -coeffs[0]  # Negative slope = error increases toward death
        
        results.per_episode_fpe = per_episode_fpe
        
        return results


def print_eval_comparison(baseline_results: FailureEvalResults, enriched_results: FailureEvalResults):
    """Pretty-print comparison between baseline and failure-enriched models."""
    print("\n" + "=" * 70)
    print("  FAILURE-CONDITIONED EVALUATION: BASELINE vs FAILURE-ENRICHED")
    print("=" * 70)
    
    print(f"\n{'Metric':<40} {'Baseline':>12} {'Enriched':>12} {'Delta':>10}")
    print("-" * 74)
    
    def row(name, b, e):
        delta = ((e - b) / b * 100) if b != 0 else 0
        sign = "+" if delta > 0 else ""
        print(f"{name:<40} {b:>12.4f} {e:>12.4f} {sign}{delta:>8.1f}%")
    
    print("\n--- Prediction Error ---")
    row("Failure frame MSE", baseline_results.failure_prediction_error, enriched_results.failure_prediction_error)
    row("Normal frame MSE", baseline_results.normal_prediction_error, enriched_results.normal_prediction_error)
    row("FPE Ratio (failure/normal)", baseline_results.fpe_ratio, enriched_results.fpe_ratio)
    
    print("\n--- Uncertainty Calibration ---")
    row("Failure frame uncertainty", baseline_results.failure_uncertainty, enriched_results.failure_uncertainty)
    row("Normal frame uncertainty", baseline_results.normal_uncertainty, enriched_results.normal_uncertainty)
    row("Uncertainty ratio", baseline_results.uncertainty_ratio, enriched_results.uncertainty_ratio)
    
    print("\n--- Recovery Accuracy ---")
    row("Recovery MSE (post-death)", baseline_results.recovery_prediction_error, enriched_results.recovery_prediction_error)
    
    print("\n--- Boundary Sharpness ---")
    row("Sharpness (error slope → death)", baseline_results.boundary_sharpness, enriched_results.boundary_sharpness)
    
    print(f"\n--- Sample Sizes ---")
    print(f"  Baseline:  {baseline_results.n_failure_frames} failure frames, {baseline_results.n_normal_frames} normal frames")
    print(f"  Enriched:  {enriched_results.n_failure_frames} failure frames, {enriched_results.n_normal_frames} normal frames")
    
    print("\n" + "=" * 70)
    
    # Key finding
    if enriched_results.fpe_ratio < baseline_results.fpe_ratio:
        improvement = (1 - enriched_results.fpe_ratio / baseline_results.fpe_ratio) * 100
        print(f"\n  KEY FINDING: Failure-enriched training reduced the FPE ratio by {improvement:.1f}%")
        print(f"  This means the model trained on failure-enriched data handles edge cases")
        print(f"  {improvement:.1f}% better relative to normal gameplay, confirming that")
        print(f"  Medal's selection bias toward dramatic clips improves world model quality")
        print(f"  at the moments that matter most.")
    else:
        print(f"\n  FINDING: FPE ratio did not improve with failure enrichment.")
        print(f"  This suggests DIAMOND may already generalize well to failure states,")
        print(f"  or that the enrichment strategy needs refinement.")
    
    print()
