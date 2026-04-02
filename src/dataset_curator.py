"""
dataset_curator.py — Creates failure-enriched vs. baseline datasets for DIAMOND training.

DIAMOND's training pipeline:
1. Collects 100k steps of gameplay (with done_on_life_loss=True)
2. Each "episode" ends at life loss → short episodes = failure events
3. Trains world model (denoiser) + reward model + actor-critic on this data

Our approach: We intercept after data collection and create two dataset variants:
- BASELINE: Standard DIAMOND data collection (unchanged)
- FAILURE-ENRICHED: Same total steps, but failure episodes duplicated to ~40% of data

Key insight from DIAMOND's codebase:
- Episodes are stored as individual .pt files in a hierarchical directory
- Dataset tracks episodes via start_idx, lengths arrays  
- BatchSampler samples episodes proportional to length (or by sample_weights)
- We can manipulate which episodes exist in the dataset to control composition

Strategy:
- After DIAMOND collects its 100k steps, we analyze the episodes
- For FAILURE_ENRICHED: we duplicate short episodes (which are failure episodes, 
  since done_on_life_loss=True means life loss → episode boundary)
- Short episodes = died quickly = failure/edge case
- Long episodes = survived longer = routine gameplay
"""

import sys
import shutil
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple

# Add DIAMOND src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "diamond" / "src"))


def analyze_episodes(dataset_dir: Path) -> List[Dict]:
    """
    Analyze all episodes in a DIAMOND dataset directory.
    
    Returns list of dicts with episode metadata, sorted by length.
    Short episodes = died quickly = failure/edge case.
    """
    from data.episode import Episode
    
    episodes = []
    episode_files = sorted(dataset_dir.rglob("*.pt"))
    
    for ep_path in episode_files:
        if ep_path.name == "info.pt":
            continue
        try:
            ep = Episode.load(ep_path)
            episodes.append({
                "path": ep_path,
                "length": len(ep),
                "return": ep.rew.sum().item(),
                "has_death": ep.end[-1].item() == 1,
                "has_positive_reward": (ep.rew > 0).any().item(),
                "num_positive_rewards": (ep.rew > 0).sum().item(),
                "num_negative_rewards": (ep.rew < 0).sum().item(),
            })
        except Exception as e:
            print(f"Warning: could not load {ep_path}: {e}")
    
    # Sort by length (short = failure episodes)
    episodes.sort(key=lambda x: x["length"])
    
    return episodes


def classify_episodes(episodes: List[Dict], percentile_threshold: float = 40.0) -> Tuple[List[Dict], List[Dict]]:
    """
    Classify episodes into "failure" and "routine" based on length.
    
    In DIAMOND with done_on_life_loss=True:
    - Short episodes = died quickly = failure/edge case
    - Long episodes = survived longer = routine gameplay
    
    Args:
        episodes: list of episode metadata dicts
        percentile_threshold: episodes shorter than this percentile are "failure"
    
    Returns:
        (failure_episodes, routine_episodes)
    """
    lengths = [ep["length"] for ep in episodes]
    threshold = np.percentile(lengths, percentile_threshold)
    
    failure_eps = [ep for ep in episodes if ep["length"] <= threshold]
    routine_eps = [ep for ep in episodes if ep["length"] > threshold]
    
    print(f"\nEpisode Classification:")
    print(f"  Length threshold (p{percentile_threshold:.0f}): {threshold:.0f} steps")
    print(f"  Failure episodes: {len(failure_eps)} (short-lived, died quickly)")
    print(f"  Routine episodes: {len(routine_eps)} (longer survival)")
    print(f"  Mean failure length: {np.mean([e['length'] for e in failure_eps]):.1f}")
    print(f"  Mean routine length: {np.mean([e['length'] for e in routine_eps]):.1f}")
    
    return failure_eps, routine_eps


def create_enriched_dataset(
    source_dir: Path,
    target_dir: Path,
    failure_episodes: List[Dict],
    routine_episodes: List[Dict],
    target_failure_ratio: float = 0.40,
    seed: int = 42,
) -> Dict:
    """
    Create a failure-enriched dataset by duplicating failure episodes.
    
    The total number of episodes increases, but the composition shifts:
    - Original: ~40% failure, ~60% routine (natural distribution)
    - Enriched: target_failure_ratio of total steps come from failure episodes
    
    We achieve this by duplicating failure episodes in the dataset directory.
    DIAMOND's Dataset class will pick them up when it scans the directory.
    
    Args:
        source_dir: Original dataset/train/ directory
        target_dir: Where to write the enriched dataset
        failure_episodes: Short episodes (failure/edge cases)
        routine_episodes: Long episodes (routine gameplay)
        target_failure_ratio: Target fraction of steps from failure episodes
        seed: Random seed
    
    Returns:
        Dict with stats about the created dataset
    """
    rng = np.random.RandomState(seed)
    
    # Calculate how many times to duplicate failure episodes
    total_failure_steps = sum(ep["length"] for ep in failure_episodes)
    total_routine_steps = sum(ep["length"] for ep in routine_episodes)
    total_steps = total_failure_steps + total_routine_steps
    
    current_failure_ratio = total_failure_steps / total_steps
    
    # To reach target ratio: failure_steps * k / (failure_steps * k + routine_steps) = target_ratio
    # k = target_ratio * routine_steps / (failure_steps * (1 - target_ratio))
    if current_failure_ratio >= target_failure_ratio:
        duplication_factor = 1
        print(f"  Failure ratio already {current_failure_ratio:.2%} >= target {target_failure_ratio:.2%}")
    else:
        duplication_factor = int(np.ceil(
            target_failure_ratio * total_routine_steps / (total_failure_steps * (1 - target_failure_ratio))
        ))
    
    print(f"\nCreating failure-enriched dataset:")
    print(f"  Original failure ratio: {current_failure_ratio:.2%}")
    print(f"  Target failure ratio: {target_failure_ratio:.2%}")
    print(f"  Failure episode duplication factor: {duplication_factor}x")
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all original episodes
    ep_id = 0
    copied_failure = 0
    copied_routine = 0
    
    # Copy routine episodes (once each)
    for ep in routine_episodes:
        dst = _get_episode_path(target_dir, ep_id)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ep["path"], dst)
        ep_id += 1
        copied_routine += 1
    
    # Copy failure episodes (duplicated)
    for dup in range(duplication_factor):
        for ep in failure_episodes:
            dst = _get_episode_path(target_dir, ep_id)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ep["path"], dst)
            ep_id += 1
            copied_failure += 1
    
    # Calculate actual ratio
    enriched_failure_steps = total_failure_steps * duplication_factor
    enriched_total_steps = enriched_failure_steps + total_routine_steps
    actual_ratio = enriched_failure_steps / enriched_total_steps
    
    stats = {
        "total_episodes": ep_id,
        "failure_episodes_original": len(failure_episodes),
        "failure_episodes_after_duplication": copied_failure,
        "routine_episodes": copied_routine,
        "duplication_factor": duplication_factor,
        "original_failure_ratio": current_failure_ratio,
        "target_failure_ratio": target_failure_ratio,
        "actual_failure_ratio": actual_ratio,
        "total_steps_original": total_steps,
        "total_steps_enriched": enriched_total_steps,
    }
    
    print(f"  Total episodes: {ep_id} (was {len(failure_episodes) + len(routine_episodes)})")
    print(f"  Actual failure ratio: {actual_ratio:.2%}")
    
    return stats


def create_baseline_dataset(source_dir: Path, target_dir: Path) -> Dict:
    """
    Create a copy of the original dataset (baseline, no modification).
    """
    print(f"\nCreating baseline dataset (copy of original)...")
    
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    
    # Count episodes
    episode_files = sorted(target_dir.rglob("*.pt"))
    episode_files = [f for f in episode_files if f.name != "info.pt"]
    
    print(f"  Copied {len(episode_files)} episodes")
    return {"total_episodes": len(episode_files), "type": "baseline"}


def _get_episode_path(base_dir: Path, episode_id: int) -> Path:
    """Replicate DIAMOND's hierarchical episode path structure."""
    n = 3
    powers = np.arange(n)
    subfolders = np.floor((episode_id % 10 ** (1 + powers)) / 10**powers) * 10**powers
    subfolders = [int(x) for x in subfolders[::-1]]
    subfolders = "/".join([f"{x:0{n - i}d}" for i, x in enumerate(subfolders)])
    return base_dir / subfolders / f"{episode_id}.pt"


def rebuild_dataset_info(dataset_dir: Path):
    """
    Rebuild DIAMOND's info.pt file for a modified dataset.
    
    DIAMOND's Dataset class uses info.pt to track:
    - num_episodes, num_steps, start_idx, lengths, counter_rew, counter_end
    
    After we modify the dataset (duplication), we need to rebuild this.
    """
    from collections import Counter
    from data.episode import Episode
    
    episode_files = sorted(dataset_dir.rglob("*.pt"))
    episode_files = [f for f in episode_files if f.name != "info.pt"]
    
    num_episodes = len(episode_files)
    lengths = []
    counter_rew = Counter()
    counter_end = Counter()
    
    for ep_path in episode_files:
        ep = Episode.load(ep_path)
        lengths.append(len(ep))
        counter_rew.update(ep.rew.sign().tolist())
        counter_end.update(ep.end.tolist())
    
    lengths = np.array(lengths, dtype=np.int64)
    start_idx = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(np.int64)
    num_steps = int(lengths.sum())
    
    state_dict = {
        "is_static": False,
        "num_episodes": num_episodes,
        "num_steps": num_steps,
        "start_idx": start_idx,
        "lengths": lengths,
        "counter_rew": counter_rew,
        "counter_end": counter_end,
    }
    
    info_path = dataset_dir / "info.pt"
    torch.save(state_dict, info_path)
    
    print(f"  Rebuilt info.pt: {num_episodes} episodes, {num_steps} steps")
    print(f"  Reward counts: {dict(counter_rew)}")
    print(f"  End counts: {dict(counter_end)}")
    
    return state_dict
