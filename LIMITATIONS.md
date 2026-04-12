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
