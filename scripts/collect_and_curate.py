#!/usr/bin/env python3
"""
Step 1: Collect gameplay data using DIAMOND's built-in collection,
then curate it into baseline vs failure-enriched datasets.

Usage:
    cd ~/diamond
    python ~/learning-from-failure/scripts/collect_and_curate.py --game BreakoutNoFrameskip-v4

This script:
1. Runs DIAMOND's data collection (100k steps using random policy)
2. Analyzes collected episodes for failure events
3. Creates two dataset variants:
   - baseline/: Original data, unchanged
   - failure_enriched/: Failure episodes duplicated for ~40% failure ratio
4. Saves analysis results to JSON
"""

import sys
import os
import json
import argparse
import shutil
from pathlib import Path

# Ensure DIAMOND's src is on path
DIAMOND_DIR = Path(os.environ.get("DIAMOND_DIR", os.path.expanduser("~/diamond")))
sys.path.insert(0, str(DIAMOND_DIR / "src"))

# Also ensure our src is on path
PROJ_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ_DIR / "src"))

import numpy as np
import torch


def collect_data(game: str, num_steps: int = 100000):
    """
    Use DIAMOND's environment to collect gameplay episodes.
    
    We use a random policy (epsilon=1.0) for initial collection,
    similar to how DIAMOND collects its initial dataset.
    """
    from envs import make_atari_env
    from data import Dataset, Episode
    
    print(f"\n{'='*60}")
    print(f"  Collecting {num_steps} steps of {game}")
    print(f"{'='*60}\n")
    
    # Create environment
    env = make_atari_env(
        num_envs=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        id=game,
        done_on_life_loss=True,  # This is key — each life loss = episode boundary
        size=64,
    )
    
    # Collect with random policy
    dataset_dir = Path("experiment_data/raw")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(dataset_dir / "train", save_on_disk=True)
    
    obs, info = env.reset()
    episode_obs = [obs]
    episode_act = []
    episode_rew = []
    episode_end = []
    episode_trunc = []
    
    steps_collected = 0
    episodes_collected = 0
    
    from tqdm import tqdm
    pbar = tqdm(total=num_steps, desc="Collecting data")
    
    while steps_collected < num_steps:
        # Random action
        action = torch.randint(0, env.num_actions, (1,), device=obs.device)
        
        next_obs, reward, end, trunc, info = env.step(action)
        
        episode_act.append(action)
        episode_rew.append(reward)
        episode_end.append(end)
        episode_trunc.append(trunc)
        
        dead = (end + trunc).clip(max=1).item()
        steps_collected += 1
        pbar.update(1)
        
        if dead:
            # Save episode
            ep = Episode(
                obs=torch.cat(episode_obs, dim=0),
                act=torch.cat(episode_act, dim=0),
                rew=torch.cat(episode_rew, dim=0),
                end=torch.cat(episode_end, dim=0),
                trunc=torch.cat(episode_trunc, dim=0),
                info={"final_observation": info.get("final_observation", next_obs)} if "final_observation" in info else {},
            )
            dataset.add_episode(ep)
            episodes_collected += 1
            
            # Reset
            obs, info = env.reset()
            episode_obs = [obs]
            episode_act = []
            episode_rew = []
            episode_end = []
            episode_trunc = []
        else:
            episode_obs.append(next_obs)
    
    # Save any remaining partial episode
    if episode_act:
        ep = Episode(
            obs=torch.cat(episode_obs, dim=0),
            act=torch.cat(episode_act, dim=0),
            rew=torch.cat(episode_rew, dim=0),
            end=torch.cat(episode_end, dim=0),
            trunc=torch.cat(episode_trunc, dim=0),
            info={},
        )
        dataset.add_episode(ep)
        episodes_collected += 1
    
    pbar.close()
    dataset.save_to_default_path()
    
    print(f"\nCollection complete:")
    print(f"  Steps: {steps_collected}")
    print(f"  Episodes: {episodes_collected}")
    print(f"  Saved to: {dataset_dir / 'train'}")
    
    return dataset_dir


def curate_datasets(raw_dir: Path, game: str):
    """
    Analyze collected data and create baseline vs failure-enriched datasets.
    """
    from dataset_curator import (
        analyze_episodes,
        classify_episodes,
        create_baseline_dataset,
        create_enriched_dataset,
        rebuild_dataset_info,
    )
    from failure_detector import FailureDetector, scan_dataset_for_failures
    
    train_dir = raw_dir / "train"
    
    print(f"\n{'='*60}")
    print(f"  Analyzing episodes for failure events")
    print(f"{'='*60}\n")
    
    # Analyze episodes
    episodes = analyze_episodes(train_dir)
    print(f"Total episodes: {len(episodes)}")
    print(f"Length range: {episodes[0]['length']} - {episodes[-1]['length']}")
    print(f"Mean length: {np.mean([e['length'] for e in episodes]):.1f}")
    print(f"Median length: {np.median([e['length'] for e in episodes]):.1f}")
    
    # Run failure detection
    detector = FailureDetector()
    scan_results = scan_dataset_for_failures(train_dir, detector)
    
    # Classify into failure vs routine
    failure_eps, routine_eps = classify_episodes(episodes, percentile_threshold=40.0)
    
    # Create baseline dataset
    print(f"\n{'='*60}")
    print(f"  Creating dataset variants")
    print(f"{'='*60}")
    
    baseline_dir = Path("experiment_data/baseline/dataset/train")
    enriched_dir = Path("experiment_data/failure_enriched/dataset/train")
    
    baseline_stats = create_baseline_dataset(train_dir, baseline_dir)
    enriched_stats = create_enriched_dataset(
        train_dir, enriched_dir,
        failure_eps, routine_eps,
        target_failure_ratio=0.40,
    )
    
    # Rebuild info.pt for enriched dataset
    print("\nRebuilding dataset metadata...")
    print("  Baseline:")
    rebuild_dataset_info(baseline_dir)
    print("  Failure-enriched:")
    rebuild_dataset_info(enriched_dir)
    
    # Also copy test dataset (same for both)
    test_dir = raw_dir / "test"
    if test_dir.exists():
        for variant in ["baseline", "failure_enriched"]:
            target_test = Path(f"experiment_data/{variant}/dataset/test")
            if target_test.exists():
                shutil.rmtree(target_test)
            shutil.copytree(test_dir, target_test)
    
    # Save analysis results
    results = {
        "game": game,
        "collection": {
            "total_episodes": len(episodes),
            "total_steps": sum(e["length"] for e in episodes),
            "mean_episode_length": float(np.mean([e["length"] for e in episodes])),
            "median_episode_length": float(np.median([e["length"] for e in episodes])),
        },
        "failure_detection": scan_results["stats"],
        "classification": {
            "failure_episodes": len(failure_eps),
            "routine_episodes": len(routine_eps),
            "mean_failure_length": float(np.mean([e["length"] for e in failure_eps])),
            "mean_routine_length": float(np.mean([e["length"] for e in routine_eps])),
        },
        "baseline": baseline_stats,
        "failure_enriched": enriched_stats,
    }
    
    results_path = Path("experiment_data/curation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"  Curation complete!")
    print(f"  Results saved to: {results_path}")
    print(f"  Baseline dataset: experiment_data/baseline/")
    print(f"  Enriched dataset: experiment_data/failure_enriched/")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Collect and curate DIAMOND training data")
    parser.add_argument("--game", type=str, default="BreakoutNoFrameskip-v4",
                       help="Atari game ID")
    parser.add_argument("--steps", type=int, default=100000,
                       help="Number of steps to collect")
    parser.add_argument("--skip-collect", action="store_true",
                       help="Skip collection, just curate existing data")
    args = parser.parse_args()
    
    os.chdir(str(DIAMOND_DIR))
    
    raw_dir = Path("experiment_data/raw")
    
    if not args.skip_collect:
        raw_dir = collect_data(args.game, args.steps)
    
    results = curate_datasets(raw_dir, args.game)
    
    print("\nReady for training! Run:")
    print(f"  GPU 1 (baseline):         python src/main.py static_dataset.path=experiment_data/baseline/dataset env.train.id={args.game} common.devices=0")
    print(f"  GPU 2 (failure-enriched):  python src/main.py static_dataset.path=experiment_data/failure_enriched/dataset env.train.id={args.game} common.devices=0")


if __name__ == "__main__":
    main()
