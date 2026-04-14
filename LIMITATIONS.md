# Limitations

## Scope
- Single environment (Atari Pong). No claim about generalization to other Atari games, 3D environments, or non-Atari domains.
- Single pretrained model (DIAMOND, Atari 100k Pong checkpoint from HuggingFace). No claim about other diffusion world models or other training regimes.
- Pong state labels are extracted via deterministic screen parsing (connected components on the ball and paddles). RAM-based labels are a planned upgrade.

## Probes (C1)
- Linear probes only. Nonlinear structure in state representations is not explored and could cause us to miss intervention-relevant directions.
- G1 as implemented is **decoding specificity**, not **intervention specificity**. Orthogonal decoding directions for two variables do not imply orthogonal causal effects when those directions are intervened on. The true G1 must come from C2 intervention-then-measure experiments. G1 as tested here is a necessary but not sufficient condition.
- ball_vx is encoded weakly (R² 0.44 at best site). Interventions on vx may be noisier than interventions on position or vy.
- Probe fits use 20k training samples drawn from a 175k-frame dataset. Frames are not independent; the temporal split mitigates but does not eliminate all dependence structure.
- The pooled-vs-flat distinction matters: flat probes at the 8×8 mid bottleneck achieve near-ceiling R² even on untrained DIAMOND, so flat-site results are a property of the architecture, not the training. Only pooled-channel results are informative about what DIAMOND learned.

## Interventions (C2, not yet run)
- Interventions will be activation-space edits at inference time. No claim about training-time representational structure or about edits in parameter space.
- Steering vectors will be derived from probe weights at channel-pooled sites (see C1 result). This inherits whatever structural assumptions the probe made (linearity, channel-wise separability).
- Temporal coherence of edited rollouts will be measured via autoregressive re-encoding; diffusion models are known to exhibit error correction that may attenuate edits regardless of whether the edit was "correct."

## Downstream (C3, not yet run)
- Downstream utility is operationalized as failure discovery with real-environment transfer. No broad robustness training claim is made in this work.
- Matched-magnitude random perturbation is the baseline. Matched-compute is the budget. Other natural baselines (targeted perturbation in pixel space, scripted rare-event seeding) are not run.

## Reproducibility
- Single random seed for the probe sweep. A seed-sensitivity check is deferred.
- HuggingFace downloads are unauthenticated; rate limits could affect reruns.

## C2 reproducibility notes (added during C2-pilot setup)

### Pong RAM ball_y = 0 is a sentinel

Pong's ALE RAM sets ball_x and ball_y to 0 when the ball is off-screen between points. Under a random policy, this occurs in ~37% of collected frames. If these are not filtered before probe fitting, the ridge probe wastes capacity on the sentinel spike and test R2 drops by ~0.12 (from 0.71 to 0.60 for ball_y at s3_mid/pooled/sigma[2]). The published `data/probe_results.csv` values assume the filter is applied. `src/steering/refit_c2_probe.py` applies `y > 1` as the filter.

### Probe direction is concentrated, not smeared

The ball_y probe direction at s3_mid/pooled/sigma[2] has ~3x more energy in the top-5 channels than would be expected under a uniform prior (top-5 weights in magnitude 0.22 to 0.38, uniform prior 1/sqrt(64) ~= 0.125). This concentration is what makes probe-direction steering plausible as a causal handle. If future probe sites show more diffuse directions (no dominant channels), the steering-via-probe-weights method will need re-justification at those sites; a diffuse probe weight vector likely picks up correlated-but-not-causal variance rather than a targeted latent axis.

### Probe-weight steering inherits linearity assumptions

The steering vector is derived from a linear ridge probe. It assumes ball_y is linearly readable from pooled channel activations at this site and timestep, which C1 confirms at R2 = 0.71. Non-linear latent structure for ball_y (e.g., the ball_y axis being a thresholded combination of channels) would be invisible to this method. If G1 passes and G2 fails (ball_y edits co-move paddle_y), one candidate explanation is that the ridge direction captures a shared spatial-position axis rather than a ball-specific axis — the paddles and ball share vertical position encoding by architecture. This would be a finding about DIAMOND's factorization, not a failure of the method.

