"""
failure_diversity.py — Failure Mode Diversity Scorer for World Model Training Data

Medal detects THAT a failure happened (kill, death, win).
This tool measures HOW DIVERSE the failure modes are.

Key insight from 1X World Model paper: "Intentionally teleoperated failures 
don't substitute because they have obvious biases." The implication: not all
failure data is equal. Diversity of failure modes matters more than quantity.

This tool:
1. Extracts frames around failure events from gameplay episodes
2. Embeds them using a pretrained vision model (no training needed)
3. Clusters the embeddings to discover failure mode categories
4. Computes a Failure Mode Diversity Score (FMDS)
5. Identifies underrepresented failure modes (blind spots)

GI could plug this into Medal's pipeline to:
- Quantify their data advantage over competitors
- Identify which games have the most diverse failure modes
- Find gaps in their training data coverage
- Prioritize data collection for underrepresented failure types

Runs on CPU. No GPU required.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

DIAMOND_SRC = Path.home() / "diamond" / "src"
if str(DIAMOND_SRC) not in sys.path:
    sys.path.insert(0, str(DIAMOND_SRC))


@dataclass
class FailureMode:
    """A discovered failure mode cluster."""
    cluster_id: int
    size: int                          # number of frames in this cluster
    fraction: float                    # fraction of all failure frames
    centroid_frame_idx: int            # index of most representative frame
    mean_episode_length: float         # avg episode length for episodes in this cluster
    mean_reward_before_death: float    # avg reward in the 10 frames before death
    description: str = ""              # auto-generated description


@dataclass  
class DiversityReport:
    """Output of the failure mode diversity analysis."""
    # Core score
    fmds: float = 0.0                 # Failure Mode Diversity Score (0-1)
    
    # Cluster info
    n_clusters: int = 0
    modes: List[FailureMode] = field(default_factory=list)
    
    # Distribution metrics
    entropy: float = 0.0              # Shannon entropy of cluster distribution
    max_entropy: float = 0.0          # Maximum possible entropy (uniform)
    evenness: float = 0.0             # entropy / max_entropy (0=all same, 1=uniform)
    
    # Coverage metrics
    coverage_radius: float = 0.0      # avg distance from points to nearest centroid
    blind_spot_ratio: float = 0.0     # fraction of embedding space not covered
    
    # Dataset stats
    total_failure_frames: int = 0
    total_episodes: int = 0
    failure_episode_fraction: float = 0.0


class FailureDiversityScorer:
    """
    Scores the diversity of failure modes in a gameplay dataset.
    
    Uses pretrained vision embeddings to discover and cluster failure types,
    then computes metrics for how diverse and well-covered the failure space is.
    """
    
    def __init__(self, n_clusters: int = 8, embedding_dim: int = 512, device: str = 'cpu'):
        """
        Args:
            n_clusters: number of failure mode clusters to discover
            embedding_dim: dimension of frame embeddings
            device: 'cpu' or 'cuda'
        """
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.device = device
        self._encoder = None
    
    def _get_encoder(self):
        """Lazy-load a pretrained image encoder for frame embedding."""
        if self._encoder is not None:
            return self._encoder
        
        try:
            # Try torchvision ResNet (always available)
            import torchvision.models as models
            import torchvision.transforms as transforms
            
            resnet = models.resnet18(weights='DEFAULT')
            # Remove classification head, keep feature extractor
            self._encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
            self._encoder.eval()
            self._encoder.to(self.device)
            
            self._transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
            ])
            print("  Using ResNet-18 encoder (512-dim embeddings)")
            
        except Exception as e:
            print(f"  Warning: Could not load ResNet encoder: {e}")
            print(f"  Falling back to raw pixel features")
            self._encoder = None
            self._transform = None
        
        return self._encoder
    
    @torch.no_grad()
    def embed_frames(self, frames: torch.Tensor) -> np.ndarray:
        """
        Embed a batch of frames into a feature space.
        
        Args:
            frames: [N, C, H, W] tensor, values in [-1, 1] (DIAMOND format)
            
        Returns:
            [N, embedding_dim] numpy array
        """
        encoder = self._get_encoder()
        
        if encoder is not None:
            # Normalize from DIAMOND's [-1, 1] to ImageNet range
            frames_01 = (frames + 1) / 2  # [-1,1] -> [0,1]
            
            # Resize and normalize for ResNet
            frames_resized = torch.nn.functional.interpolate(
                frames_01, size=(224, 224), mode='bilinear', align_corners=False
            )
            frames_norm = self._transform(frames_resized) if self._transform else frames_resized
            
            # Batch encode
            batch_size = 64
            embeddings = []
            for i in range(0, len(frames_norm), batch_size):
                batch = frames_norm[i:i+batch_size].to(self.device)
                emb = encoder(batch).squeeze(-1).squeeze(-1)
                embeddings.append(emb.cpu().numpy())
            
            return np.concatenate(embeddings, axis=0)
        else:
            # Fallback: flatten and PCA
            flat = frames.reshape(len(frames), -1).numpy()
            # Simple dimensionality reduction
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(self.embedding_dim, flat.shape[1], len(flat)))
            return pca.fit_transform(flat)
    
    def extract_failure_frames(self, episodes: list, window: int = 5) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Extract frames around failure events from episodes.
        
        Args:
            episodes: list of DIAMOND Episode objects
            window: number of frames before death to extract
            
        Returns:
            failure_frames: [N, C, H, W] tensor
            frame_metadata: list of dicts with episode info
        """
        from failure_detector import FailureDetector
        detector = FailureDetector(near_death_window=window)
        
        all_frames = []
        all_metadata = []
        
        for ep_idx, episode in enumerate(episodes):
            T = len(episode)
            events = detector.detect_from_episode(episode, ep_idx)
            death_events = [e for e in events if e.event_type == "life_loss"]
            
            for event in death_events:
                # Extract frames leading up to death
                start = max(0, event.frame_idx - window)
                end = min(T, event.frame_idx + 1)
                
                # Use the frame just before death as the representative frame
                frame_idx = max(0, event.frame_idx - 1)
                frame = episode.obs[frame_idx]
                all_frames.append(frame)
                
                # Metadata for this failure
                reward_window = episode.rew[start:end].sum().item()
                all_metadata.append({
                    'episode_id': ep_idx,
                    'frame_idx': frame_idx,
                    'death_frame': event.frame_idx,
                    'episode_length': T,
                    'reward_before_death': reward_window,
                    'total_episode_return': episode.rew.sum().item(),
                })
        
        if not all_frames:
            return torch.tensor([]), []
        
        return torch.stack(all_frames), all_metadata
    
    def cluster_failures(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster failure frame embeddings to discover failure mode categories.
        
        Returns:
            labels: [N] cluster assignments
            centroids: [K, D] cluster centers
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_norm = scaler.fit_transform(embeddings)
        
        # Determine optimal number of clusters (up to n_clusters)
        n = min(self.n_clusters, len(embeddings) // 3)  # need at least 3 per cluster
        n = max(2, n)
        
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_norm)
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        
        return labels, centroids
    
    def compute_diversity_score(
        self, 
        labels: np.ndarray, 
        embeddings: np.ndarray,
        centroids: np.ndarray,
        metadata: List[Dict],
    ) -> DiversityReport:
        """
        Compute the Failure Mode Diversity Score and related metrics.
        
        FMDS = evenness × coverage_quality
        
        - Evenness: How uniformly distributed are failures across modes?
          (Shannon entropy normalized by max entropy)
        - Coverage: How well do clusters cover the embedding space?
          (inverse of mean distance to nearest centroid)
        """
        n_clusters = len(centroids)
        cluster_sizes = Counter(labels)
        
        # Shannon entropy of cluster distribution
        total = len(labels)
        probs = np.array([cluster_sizes[i] / total for i in range(n_clusters)])
        probs = probs[probs > 0]  # remove empty clusters
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(n_clusters)
        evenness = entropy / max_entropy if max_entropy > 0 else 0
        
        # Coverage: mean distance to nearest centroid (lower = better coverage)
        from sklearn.metrics import pairwise_distances
        dists = pairwise_distances(embeddings, centroids)
        min_dists = dists.min(axis=1)
        coverage_radius = float(np.mean(min_dists))
        
        # Normalize coverage to [0, 1] (1 = good coverage)
        max_dist = float(np.max(min_dists))
        coverage_quality = 1.0 - (coverage_radius / max_dist) if max_dist > 0 else 1.0
        
        # FMDS = evenness × coverage_quality
        fmds = evenness * coverage_quality
        
        # Build failure mode descriptions
        modes = []
        for k in range(n_clusters):
            mask = labels == k
            cluster_meta = [m for m, l in zip(metadata, labels) if l == k]
            
            if not cluster_meta:
                continue
            
            # Find the frame closest to centroid (most representative)
            cluster_embeddings = embeddings[mask]
            centroid = centroids[k]
            dists_to_centroid = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            centroid_idx = np.where(mask)[0][np.argmin(dists_to_centroid)]
            
            mode = FailureMode(
                cluster_id=k,
                size=int(mask.sum()),
                fraction=float(mask.sum()) / total,
                centroid_frame_idx=int(centroid_idx),
                mean_episode_length=float(np.mean([m['episode_length'] for m in cluster_meta])),
                mean_reward_before_death=float(np.mean([m['reward_before_death'] for m in cluster_meta])),
            )
            
            # Auto-describe based on characteristics
            if mode.mean_episode_length < 30:
                mode.description = "Quick death (short episode)"
            elif mode.mean_reward_before_death > 0:
                mode.description = "Death after scoring (mid-game failure)"
            elif mode.mean_episode_length > 80:
                mode.description = "Late-game death (long survival)"
            else:
                mode.description = "Standard death"
            
            modes.append(mode)
        
        # Sort by size
        modes.sort(key=lambda m: m.size, reverse=True)
        
        # Blind spot ratio: fraction of "space" covered by smallest clusters
        small_cluster_threshold = total * 0.05  # clusters with < 5% of data
        blind_spots = sum(1 for m in modes if m.size < small_cluster_threshold)
        blind_spot_ratio = blind_spots / n_clusters if n_clusters > 0 else 0
        
        return DiversityReport(
            fmds=fmds,
            n_clusters=n_clusters,
            modes=modes,
            entropy=entropy,
            max_entropy=max_entropy,
            evenness=evenness,
            coverage_radius=coverage_radius,
            blind_spot_ratio=blind_spot_ratio,
            total_failure_frames=total,
        )
    
    def score_dataset(self, episodes: list, window: int = 5) -> DiversityReport:
        """
        End-to-end: score the failure mode diversity of a dataset.
        
        Args:
            episodes: list of DIAMOND Episode objects
            window: frames before death to consider
            
        Returns:
            DiversityReport with all metrics
        """
        print("  Step 1: Extracting failure frames...")
        frames, metadata = self.extract_failure_frames(episodes, window)
        
        if len(frames) == 0:
            print("  No failure frames found!")
            return DiversityReport()
        
        print(f"  Found {len(frames)} failure frames from {len(episodes)} episodes")
        
        print("  Step 2: Embedding frames...")
        embeddings = self.embed_frames(frames)
        print(f"  Embedding shape: {embeddings.shape}")
        
        print("  Step 3: Clustering failure modes...")
        labels, centroids = self.cluster_failures(embeddings)
        print(f"  Found {len(set(labels))} failure mode clusters")
        
        print("  Step 4: Computing diversity score...")
        report = self.compute_diversity_score(labels, embeddings, centroids, metadata)
        report.total_episodes = len(episodes)
        report.failure_episode_fraction = len(frames) / max(len(episodes), 1)
        
        return report


def print_diversity_report(report: DiversityReport, dataset_name: str = "Dataset"):
    """Pretty-print a diversity report."""
    print(f"\n{'='*60}")
    print(f"  FAILURE MODE DIVERSITY REPORT: {dataset_name}")
    print(f"{'='*60}")
    
    print(f"\n  Failure Mode Diversity Score (FMDS): {report.fmds:.3f}")
    print(f"  (0 = all failures identical, 1 = maximally diverse)")
    
    print(f"\n  --- Distribution ---")
    print(f"  Failure frames:     {report.total_failure_frames}")
    print(f"  Episodes analyzed:  {report.total_episodes}")
    print(f"  Clusters found:     {report.n_clusters}")
    print(f"  Entropy:            {report.entropy:.3f} / {report.max_entropy:.3f}")
    print(f"  Evenness:           {report.evenness:.3f}")
    print(f"  Blind spot ratio:   {report.blind_spot_ratio:.2%}")
    
    print(f"\n  --- Failure Modes ---")
    for mode in report.modes:
        bar = '█' * int(mode.fraction * 40)
        print(f"  Cluster {mode.cluster_id}: {bar} {mode.size} frames ({mode.fraction:.1%})")
        print(f"    Avg episode length: {mode.mean_episode_length:.0f} | "
              f"Reward before death: {mode.mean_reward_before_death:.1f} | "
              f"{mode.description}")
    
    print(f"\n{'='*60}")


def compare_diversity(report_a: DiversityReport, report_b: DiversityReport,
                      name_a: str = "Dataset A", name_b: str = "Dataset B"):
    """Compare diversity scores between two datasets."""
    print(f"\n{'='*60}")
    print(f"  DIVERSITY COMPARISON: {name_a} vs {name_b}")
    print(f"{'='*60}")
    
    print(f"\n  {'Metric':<30} {name_a:>15} {name_b:>15}")
    print(f"  {'-'*60}")
    print(f"  {'FMDS Score':<30} {report_a.fmds:>15.3f} {report_b.fmds:>15.3f}")
    print(f"  {'Clusters':<30} {report_a.n_clusters:>15d} {report_b.n_clusters:>15d}")
    print(f"  {'Evenness':<30} {report_a.evenness:>15.3f} {report_b.evenness:>15.3f}")
    print(f"  {'Entropy':<30} {report_a.entropy:>15.3f} {report_b.entropy:>15.3f}")
    print(f"  {'Blind Spot Ratio':<30} {report_a.blind_spot_ratio:>14.1%} {report_b.blind_spot_ratio:>14.1%}")
    print(f"  {'Failure Frames':<30} {report_a.total_failure_frames:>15d} {report_b.total_failure_frames:>15d}")
    
    if report_a.fmds > report_b.fmds:
        diff = (report_a.fmds - report_b.fmds) / report_b.fmds * 100 if report_b.fmds > 0 else 0
        print(f"\n  {name_a} has {diff:.1f}% higher failure mode diversity")
    else:
        diff = (report_b.fmds - report_a.fmds) / report_a.fmds * 100 if report_a.fmds > 0 else 0
        print(f"\n  {name_b} has {diff:.1f}% higher failure mode diversity")
    
    print(f"\n{'='*60}")
