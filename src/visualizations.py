"""
visualizations.py — Publication-quality figures for the failure evaluation.

Generates all key plots for the project writeup:
1. FPE comparison bar chart (baseline vs enriched)
2. Uncertainty calibration curves
3. Failure boundary sharpness plot
4. Failure taxonomy breakdown
5. Agent return comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional

# Style
sns.set_theme(style="whitegrid", font_scale=1.2)
COLORS = {
    'baseline': '#4C72B0',
    'enriched': '#DD8452',
    'failure': '#C44E52',
    'normal': '#55A868',
}


def plot_fpe_comparison(baseline_results, enriched_results, save_path: Path):
    """
    Bar chart comparing Failure Prediction Error between baseline and enriched models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: MSE comparison
    ax = axes[0]
    x = np.arange(2)
    width = 0.35
    baseline_vals = [baseline_results.failure_prediction_error, baseline_results.normal_prediction_error]
    enriched_vals = [enriched_results.failure_prediction_error, enriched_results.normal_prediction_error]
    
    ax.bar(x - width/2, baseline_vals, width, label='Baseline', color=COLORS['baseline'])
    ax.bar(x + width/2, enriched_vals, width, label='Failure-Enriched', color=COLORS['enriched'])
    ax.set_xticks(x)
    ax.set_xticklabels(['Failure\nFrames', 'Normal\nFrames'])
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Prediction Error by Frame Type')
    ax.legend()
    
    # Panel 2: FPE Ratio
    ax = axes[1]
    ratios = [baseline_results.fpe_ratio, enriched_results.fpe_ratio]
    bars = ax.bar(['Baseline', 'Failure-\nEnriched'], ratios, 
                   color=[COLORS['baseline'], COLORS['enriched']], width=0.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal performance')
    ax.set_ylabel('FPE Ratio (failure MSE / normal MSE)')
    ax.set_title('Failure Prediction Error Ratio')
    ax.legend()
    
    # Add value labels
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Panel 3: Per-episode FPE distribution
    ax = axes[2]
    if baseline_results.per_episode_fpe and enriched_results.per_episode_fpe:
        ax.hist(baseline_results.per_episode_fpe, bins=20, alpha=0.6, 
                label='Baseline', color=COLORS['baseline'])
        ax.hist(enriched_results.per_episode_fpe, bins=20, alpha=0.6,
                label='Failure-Enriched', color=COLORS['enriched'])
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Per-Episode FPE Ratio')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Per-Episode FPE')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / 'fpe_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'fpe_comparison.png'}")


def plot_uncertainty_calibration(baseline_results, enriched_results, save_path: Path):
    """
    Compare uncertainty calibration between models.
    Well-calibrated model = higher uncertainty at harder (failure) frames.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Scatter: baseline
    categories = ['Normal', 'Failure']
    baseline_unc = [baseline_results.normal_uncertainty, baseline_results.failure_uncertainty]
    enriched_unc = [enriched_results.normal_uncertainty, enriched_results.failure_uncertainty]
    baseline_err = [baseline_results.normal_prediction_error, baseline_results.failure_prediction_error]
    enriched_err = [enriched_results.normal_prediction_error, enriched_results.failure_prediction_error]
    
    ax.scatter(baseline_err, baseline_unc, s=200, marker='o', color=COLORS['baseline'],
              label='Baseline', zorder=5, edgecolors='black', linewidth=1)
    ax.scatter(enriched_err, enriched_unc, s=200, marker='s', color=COLORS['enriched'],
              label='Failure-Enriched', zorder=5, edgecolors='black', linewidth=1)
    
    # Annotate points
    for i, cat in enumerate(categories):
        ax.annotate(f'{cat}\n(Baseline)', (baseline_err[i], baseline_unc[i]),
                   textcoords="offset points", xytext=(10, 10), fontsize=9, alpha=0.7)
        ax.annotate(f'{cat}\n(Enriched)', (enriched_err[i], enriched_unc[i]),
                   textcoords="offset points", xytext=(10, -15), fontsize=9, alpha=0.7)
    
    # Perfect calibration line
    all_err = baseline_err + enriched_err
    if all_err:
        min_e, max_e = min(all_err), max(all_err)
        ax.plot([min_e, max_e], [min_e, max_e], 'k--', alpha=0.3, label='Perfect calibration')
    
    ax.set_xlabel('Prediction Error (MSE)')
    ax.set_ylabel('Prediction Uncertainty (Diffusion Variance)')
    ax.set_title('Uncertainty Calibration: Does the Model Know What It Doesn\'t Know?')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / 'uncertainty_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'uncertainty_calibration.png'}")


def plot_boundary_sharpness(baseline_results, enriched_results, save_path: Path):
    """
    Plot prediction error vs distance to next failure event.
    Shows how world model quality degrades as we approach a failure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if baseline_results.boundary_errors:
        dists_b = sorted(baseline_results.boundary_errors.keys())
        errors_b = [baseline_results.boundary_errors[d] for d in dists_b]
        ax.plot(dists_b, errors_b, 'o-', color=COLORS['baseline'], 
                label=f'Baseline (sharpness={baseline_results.boundary_sharpness:.4f})',
                markersize=6, linewidth=2)
    
    if enriched_results.boundary_errors:
        dists_e = sorted(enriched_results.boundary_errors.keys())
        errors_e = [enriched_results.boundary_errors[d] for d in dists_e]
        ax.plot(dists_e, errors_e, 's-', color=COLORS['enriched'],
                label=f'Failure-Enriched (sharpness={enriched_results.boundary_sharpness:.4f})',
                markersize=6, linewidth=2)
    
    ax.axvline(x=0, color=COLORS['failure'], linestyle='--', alpha=0.5, label='Death event')
    ax.set_xlabel('Frames Until Next Death Event')
    ax.set_ylabel('Mean Prediction Error (MSE)')
    ax.set_title('Failure Boundary Sharpness: How Error Evolves Approaching Death')
    ax.legend()
    ax.invert_xaxis()  # Death at left, far from death at right
    
    plt.tight_layout()
    plt.savefig(save_path / 'boundary_sharpness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'boundary_sharpness.png'}")


def plot_failure_taxonomy(episode_stats, save_path: Path):
    """
    Breakdown of failure event types detected in the dataset.
    Shows what kinds of failures the detector found.
    """
    from collections import Counter
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Episode length distribution colored by failure classification
    ax = axes[0]
    lengths = [ep['length'] for ep in episode_stats]
    failure_mask = [ep.get('failure_density', 0) > 0.3 for ep in episode_stats]
    
    failure_lengths = [l for l, f in zip(lengths, failure_mask) if f]
    routine_lengths = [l for l, f in zip(lengths, failure_mask) if not f]
    
    ax.hist(routine_lengths, bins=30, alpha=0.6, label='Routine episodes', color=COLORS['normal'])
    ax.hist(failure_lengths, bins=30, alpha=0.6, label='Failure episodes', color=COLORS['failure'])
    ax.set_xlabel('Episode Length (steps)')
    ax.set_ylabel('Count')
    ax.set_title('Episode Length Distribution\n(Short = died quickly = failure)')
    ax.legend()
    
    # Panel 2: Event type breakdown
    ax = axes[1]
    type_counts = Counter()
    for ep in episode_stats:
        if 'event_types' in ep:
            type_counts.update(ep['event_types'])
    
    if type_counts:
        types = list(type_counts.keys())
        counts = [type_counts[t] for t in types]
        ax.barh(types, counts, color=COLORS['failure'])
        ax.set_xlabel('Count')
        ax.set_title('Failure Event Types Detected')
    else:
        ax.text(0.5, 0.5, 'No event type data', transform=ax.transAxes,
                ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(save_path / 'failure_taxonomy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'failure_taxonomy.png'}")


def plot_agent_returns(baseline_returns, enriched_returns, save_path: Path):
    """
    Compare agent returns (the standard metric) alongside our failure metrics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Return distributions
    ax = axes[0]
    ax.hist(baseline_returns, bins=20, alpha=0.6, label='Baseline', color=COLORS['baseline'])
    ax.hist(enriched_returns, bins=20, alpha=0.6, label='Failure-Enriched', color=COLORS['enriched'])
    ax.set_xlabel('Episode Return')
    ax.set_ylabel('Count')
    ax.set_title('Agent Return Distribution')
    ax.legend()
    
    # Panel 2: Mean comparison
    ax = axes[1]
    means = [np.mean(baseline_returns), np.mean(enriched_returns)]
    stds = [np.std(baseline_returns), np.std(enriched_returns)]
    bars = ax.bar(['Baseline', 'Failure-\nEnriched'], means, yerr=stds,
                   color=[COLORS['baseline'], COLORS['enriched']], width=0.5,
                   capsize=5)
    ax.set_ylabel('Mean Episode Return')
    ax.set_title('Agent Performance Comparison')
    
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / 'agent_returns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path / 'agent_returns.png'}")


def generate_all_figures(baseline_results, enriched_results, 
                         episode_stats=None, baseline_returns=None, 
                         enriched_returns=None, save_dir='results/figures'):
    """Generate all figures for the project."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating figures...")
    plot_fpe_comparison(baseline_results, enriched_results, save_path)
    plot_uncertainty_calibration(baseline_results, enriched_results, save_path)
    plot_boundary_sharpness(baseline_results, enriched_results, save_path)
    
    if episode_stats:
        plot_failure_taxonomy(episode_stats, save_path)
    if baseline_returns is not None and enriched_returns is not None:
        plot_agent_returns(baseline_returns, enriched_returns, save_path)
    
    print(f"\nAll figures saved to {save_path}/")
