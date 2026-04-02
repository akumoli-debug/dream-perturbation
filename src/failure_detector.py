"""
failure_detector.py — Detects failure events in Atari gameplay episodes.

Works directly with DIAMOND's Episode format (obs, act, rew, end, trunc, info).

A failure event is:
- Life loss: the `end` flag is set to 1 during training (DIAMOND uses done_on_life_loss=True)
- Terminal: episode truly ends (end=1 in test, or final trunc=1)
- Reward drought: long consecutive stretches with zero reward
- Near-death: frames just BEFORE a life loss (the approach to failure)

This mimics Medal's natural selection bias — gamers clip dramatic/failure moments.
We build the synthetic equivalent for controlled experiments.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set
from pathlib import Path


@dataclass
class FailureEvent:
    """A detected failure event in an episode."""
    episode_id: int
    frame_idx: int              # The frame where the event occurs
    event_type: str             # "life_loss", "terminal", "near_death", "reward_drought"
    severity: float             # 0.0–1.0
    context_window: Tuple[int, int]  # (start, end) inclusive frame range


class FailureDetector:
    """
    Detects failure events in DIAMOND Episode objects.
    
    DIAMOND's training env uses done_on_life_loss=True, meaning
    each life loss creates a separate episode that ends with end=1.
    So every episode boundary in training data IS a failure event.
    
    For richer analysis, we also detect:
    - Reward droughts (extended zero-reward periods)
    - Near-death sequences (frames approaching the end of an episode)
    """
    
    def __init__(
        self,
        near_death_window: int = 10,   # frames before death to mark as "near-death"
        drought_threshold: int = 50,    # consecutive zero-reward frames = drought
        post_event_window: int = 3,     # frames after event to include
    ):
        self.near_death_window = near_death_window
        self.drought_threshold = drought_threshold
        self.post_event_window = post_event_window
    
    def detect_from_episode(self, episode, episode_id: int = 0) -> List[FailureEvent]:
        """
        Detect failure events from a DIAMOND Episode object.
        
        Args:
            episode: DIAMOND Episode with .obs, .act, .rew, .end, .trunc
            episode_id: ID for tracking
            
        Returns:
            List of FailureEvent objects
        """
        events = []
        T = len(episode)
        rew = episode.rew.cpu().numpy()
        end = episode.end.cpu().numpy()
        trunc = episode.trunc.cpu().numpy()
        
        # 1. Terminal events (life loss or game over)
        # In DIAMOND training, every episode ends with end=1 (life loss) or trunc=1
        for t in range(T):
            if end[t] == 1:
                events.append(FailureEvent(
                    episode_id=episode_id,
                    frame_idx=t,
                    event_type="life_loss",
                    severity=1.0,
                    context_window=(max(0, t - self.near_death_window), min(T - 1, t + self.post_event_window))
                ))
        
        # 2. Near-death frames (approach to failure)
        for t in range(T):
            if end[t] == 1:
                start = max(0, t - self.near_death_window)
                for nd_t in range(start, t):
                    # Severity increases as we get closer to death
                    distance_to_death = t - nd_t
                    severity = 1.0 - (distance_to_death / self.near_death_window)
                    events.append(FailureEvent(
                        episode_id=episode_id,
                        frame_idx=nd_t,
                        event_type="near_death",
                        severity=severity,
                        context_window=(max(0, nd_t - 2), min(T - 1, nd_t + 2))
                    ))
        
        # 3. Reward droughts
        drought_start = None
        drought_len = 0
        for t in range(T):
            if rew[t] == 0:
                if drought_start is None:
                    drought_start = t
                drought_len += 1
            else:
                if drought_len >= self.drought_threshold:
                    events.append(FailureEvent(
                        episode_id=episode_id,
                        frame_idx=drought_start,
                        event_type="reward_drought",
                        severity=min(1.0, drought_len / (self.drought_threshold * 3)),
                        context_window=(drought_start, t - 1)
                    ))
                drought_start = None
                drought_len = 0
        
        # Handle drought at end of episode
        if drought_len >= self.drought_threshold and drought_start is not None:
            events.append(FailureEvent(
                episode_id=episode_id,
                frame_idx=drought_start,
                event_type="reward_drought",
                severity=min(1.0, drought_len / (self.drought_threshold * 3)),
                context_window=(drought_start, T - 1)
            ))
        
        return events
    
    def get_failure_frame_set(self, events: List[FailureEvent]) -> Set[int]:
        """Get the set of all frame indices that are part of failure events."""
        frames = set()
        for event in events:
            for t in range(event.context_window[0], event.context_window[1] + 1):
                frames.add(t)
        return frames
    
    def compute_stats(self, events: List[FailureEvent], total_frames: int) -> Dict:
        """Compute summary statistics about detected failure events."""
        failure_frames = self.get_failure_frame_set(events)
        
        by_type = {}
        for event in events:
            by_type.setdefault(event.event_type, []).append(event)
        
        return {
            "total_events": len(events),
            "total_frames": total_frames,
            "failure_frames": len(failure_frames),
            "failure_density": len(failure_frames) / total_frames if total_frames > 0 else 0,
            "events_by_type": {k: len(v) for k, v in by_type.items()},
            "mean_severity": np.mean([e.severity for e in events]) if events else 0,
        }


def scan_dataset_for_failures(dataset_path: Path, detector: FailureDetector = None) -> Dict:
    """
    Scan a DIAMOND dataset directory for failure events.
    
    Args:
        dataset_path: Path to dataset/train/ directory
        detector: FailureDetector instance (creates default if None)
    
    Returns:
        Dict with per-episode events and aggregate stats
    """
    if detector is None:
        detector = FailureDetector()
    
    from data.episode import Episode
    
    all_events = []
    total_frames = 0
    episode_stats = []
    
    # Walk the hierarchical directory structure DIAMOND uses
    episode_files = sorted(dataset_path.rglob("*.pt"))
    if not episode_files:
        print(f"No episode files found in {dataset_path}")
        return {"events": [], "stats": {}}
    
    print(f"Scanning {len(episode_files)} episodes for failure events...")
    
    for ep_idx, ep_path in enumerate(episode_files):
        if ep_path.name == "info.pt":
            continue
        try:
            episode = Episode.load(ep_path)
            events = detector.detect_from_episode(episode, episode_id=ep_idx)
            all_events.extend(events)
            total_frames += len(episode)
            
            episode_stats.append({
                "episode_id": ep_idx,
                "length": len(episode),
                "return": episode.rew.sum().item(),
                "num_events": len(events),
                "failure_density": len(detector.get_failure_frame_set(events)) / len(episode),
            })
        except Exception as e:
            print(f"  Warning: failed to load {ep_path}: {e}")
    
    stats = detector.compute_stats(all_events, total_frames)
    
    print(f"\nFailure Detection Summary:")
    print(f"  Episodes scanned: {len(episode_stats)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Failure events: {stats['total_events']}")
    print(f"  Failure density: {stats['failure_density']:.2%}")
    print(f"  Events by type: {stats['events_by_type']}")
    
    return {
        "events": all_events,
        "episode_stats": episode_stats,
        "stats": stats,
    }
