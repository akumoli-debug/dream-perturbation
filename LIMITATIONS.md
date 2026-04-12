# Limitations

- Single environment (Atari Pong). No claim about generalization to other Atari games, 3D environments, or non-Atari domains.
- Linear probes only. Nonlinear structure in state representations is not explored and could cause us to miss intervention-relevant directions.
- Interventions are activation-space edits at inference time. No claim about training-time representational structure or about edits in parameter space.
- Downstream utility is operationalized as failure discovery with real-environment transfer. No broad robustness training claim is made in this work.
- Pong state labels are extracted via deterministic screen parsing (connected components on the ball and paddles). RAM-based labels are a planned upgrade.
