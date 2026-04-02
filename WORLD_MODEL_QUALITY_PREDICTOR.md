# World Model Quality Predictor: Design Document

## The Problem

Training an RL agent inside a world model takes hours to days. GI needs to evaluate hundreds of architecture variants, data mixtures, and hyperparameter configurations. Currently, the only way to know if a world model is good is to train an RL agent inside it and measure the agent's score. This is the evaluation bottleneck.

**The question: Can we predict how well an RL agent will perform inside a world model, without actually running RL?**

## Why This Is Hard

WorldArena (Feb 2026) showed that standard visual quality metrics (FID, FVD, SSIM, PSNR) do NOT correlate with downstream task performance. So simple "how pretty are the frames" doesn't work.

The reason: A world model can produce beautiful frames but have:
- Wrong physics (ball passes through paddle)
- Wrong reward predictions (agent gets reward for nothing)
- Compounding error (drifts out of distribution over 100+ steps)
- Mode collapse (always predicts the same "safe" transition)
- Poor action sensitivity (different actions → same outcome)

## Our Approach: 7 Proxy Features That Capture What Matters

For each pretrained DIAMOND world model (26 games), we compute 7 features that capture different aspects of world model quality. We then correlate these with DIAMOND's published agent scores to find which features predict RL performance.

### Feature 1: Single-Step Prediction Error (baseline metric)
- MSE between world model's predicted next frame and actual next frame
- Computed on held-out test episodes
- This is what everyone measures. We include it as baseline.

### Feature 2: Multi-Step Rollout Stability (compounding error)
- Generate 50-step rollouts from the world model (autoregressive)
- Measure how fast prediction error grows: MSE at step 1, 5, 10, 20, 50
- Compute the "error doubling time" — how many steps before error doubles
- A stable world model has slow error growth; an unstable one explodes

### Feature 3: Action Sensitivity (does the model respond to different actions?)
- From the same state, predict next frame for EVERY possible action
- Measure variance across predictions: high variance = action-sensitive
- A world model where all actions produce the same output is useless for RL
- Compute: mean pairwise MSE between predictions for different actions

### Feature 4: Stochasticity Calibration (does the model capture uncertainty?)
- Run the same (state, action) through the diffusion model N times
- Measure variance of outputs (diffusion stochasticity)
- Compare to actual next-frame variance in the real environment
- A well-calibrated model's uncertainty should correlate with true uncertainty

### Feature 5: Reward Prediction Accuracy
- DIAMOND has a separate reward/end predictor (rew_end_model)
- Measure accuracy of reward predictions on test data
- Also measure: does the model correctly predict WHEN episodes end?
- Bad reward prediction → RL agent gets wrong training signal → poor policy

### Feature 6: Visual Detail Preservation (DIAMOND-specific)
- DIAMOND paper showed "visual details matter" — fine details predict RL score
- Compute LPIPS (perceptual similarity) between predicted and actual frames
- Also: measure high-frequency energy in predicted frames (do details survive?)
- Blurry world model → agent can't distinguish important visual cues

### Feature 7: State Space Coverage
- Embed world model rollout frames with ResNet
- Embed real environment frames with ResNet
- Measure KL divergence / MMD between the two distributions
- If the world model generates states the real env never visits (or vice versa), RL trained inside it will fail when transferred to reality

## The Data

We have 26 DIAMOND pretrained models, each trained on a different Atari game with 5 random seeds. Published mean scores available.

For each game:
1. Download pretrained model from HuggingFace
2. Collect 50 real environment episodes (random policy, ~10k steps)
3. Compute all 7 features using the pretrained world model
4. Record the published DIAMOND agent score (mean of 5 seeds)

## Analysis

With 26 data points (games) and 7 features:
1. Individual correlations: which features predict HNS?
2. Multiple regression: which combination of features best predicts HNS?
3. Leave-one-out cross-validation: how well does the predictor generalize?
4. Feature importance ranking: which features matter most?

## Why This Matters for GI

If even 2-3 of these features correlate with agent performance, GI could:
1. Evaluate world model architectures 100x faster (minutes vs days)
2. Do hyperparameter search using proxy metrics
3. Compare data mixtures without running full RL training
4. Monitor training progress with leading indicators

## Compute Requirements

- No GPU needed — all inference on pretrained models, CPU is fine
- ~26 games × 30 min each = ~13 hours total
- Can be parallelized across CPU cores
- Cost: $0

## What Makes This Non-Trivial

1. Nobody has done this for game world models (WorldArena did it for robotics, found nothing works)
2. The feature design requires understanding WHY RL succeeds or fails inside world models
3. Multi-step rollout stability and action sensitivity are novel metrics specific to interactive world models
4. If we find a working predictor, it has immediate practical value for any world model lab
5. Even null results (no features predict) are publishable — it characterizes the hardness of the evaluation problem
