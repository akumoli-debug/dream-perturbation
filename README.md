# Dream Perturbation

**Thesis.** Diffusion world models may contain editable internal directions corresponding to environment state. We test whether interventions on these directions generate selective, temporally coherent counterfactual rollouts, and whether those rollouts provide more useful training signal than untargeted perturbations.

## Claim hierarchy

- **C1** — Internal directions in DIAMOND's U-Net encode environment variables in an intervention-relevant way.
- **C2** — Interventions on those directions produce selective and temporally coherent counterfactual rollouts.
- **C3** — Targeted rollouts improve downstream failure discovery over matched-magnitude random perturbation, and those failures transfer to the real Atari environment.

Each claim is independently publishable. If C3 fails, C1+C2 stand as a mechanistic interpretability result. If C2 fails, C1 stands as a probing result. If C1 fails, the negative result is itself a finding about diffusion world model internals.

## Experiments

### C1 — Probe results (2026-04-12)

**Setup.** 24,000 Pong frames, 3 denoising timesteps (σ=5.0, 0.28, 0.002), 6 probe sites across the U-Net (conv_in, 4 down/up block outputs, mid bottleneck). Ridge probes on two representations: pooled (64-dim, mean over spatial axes) and flat (4096-dim, full spatial footprint at 8×8 sites). Train/test split is temporal (last 20% by frame index held out). Seven target variables: ball position/velocity in x/y, both paddle y positions, and collision_imminent.

**Gate G1 (decoding specificity).** A site passes if (i) ≥3 probes achieve test R² > 0.1 on their own target and (ii) the median ratio of on-target R² to max off-target R² exceeds 2. G1 passes at every mid/up site except s5 (final 64×64 pre-output, where channel-pooled representations have collapsed back to spatial encoding) and s0 (pre-U-Net conv).

**The sanity check that reframed C1.** An identical sweep on a randomly-initialized DIAMOND (same architecture, no pretraining) also passes G1 at flat-mid with R² ≈ 0.95 on ball/paddle position. This is not a bug. A 4096-dim spatial representation of a 64×64 Pong frame is trivially decodable because Pong is visually simple and random conv features preserve spatial structure. **The flat-mid result is a finding about the architecture, not about DIAMOND's training.**

Training *does* show up, but in the pooled (channel-only) representation:

| site | pooled s[2] diag R² (trained) | pooled s[2] diag R² (untrained) |
|---|---|---|
| s1_d_stage1 | 0.25 | no signal |
| s2_d_stage3 | 0.43 | 0.28 |
| s3_mid | 0.53 | 0.24 |
| s4_u_stage1 | 0.45 | 0.22 (fails gate) |

So the actual C1 finding is: **DIAMOND's pretraining organizes environment variables along channel dimensions such that they are recoverable by low-dimensional linear probes.** The same variables are extractable from untrained activations only by probes with access to the full spatial representation.

**Consequence for C2.** Steering vectors must be defined in pooled channel space at a mid or early-up site, not in flat spatial space. Flat-space interventions would be editing random-feature structure, not learned structure.

**Variable-specific ceilings (best site: s3_mid flat s[2]).** ball_y 0.994, left_paddle_y 0.996, right_paddle_y 0.987, collision_imminent 0.999 (AUC), ball_x 0.915, ball_vy 0.856, **ball_vx 0.437**. The x-velocity ceiling is a known obstacle for C2: an intervention on vx will have a noisier steering direction than interventions on position or vy.

**Negative controls.** Shuffled labels collapse every probe to R² ≈ 0 on both trained and untrained activations (rules out ridge overfitting). Temporal held-out split with a frame-index gap rules out conditioning-window leakage.

Artifacts: `data/probe_results*.csv`, `data/gate_summary*.csv`, `data/specificity_matrix*.pt` (four conditions: trained, untrained, trained+shuffled, untrained+shuffled).


- `experiments/c1_probe_map/` — layer × denoising-timestep probe sweep + intervention specificity matrix.
- `experiments/c2_counterfactual/` — selectivity over rollout horizon, persistence, re-encoding stability, playability, matched-random baseline.
- `experiments/c3_failure_discovery/` — targeted vs matched-random edit mining, real-environment transfer.

## Gates

- **G1 (specificity).** Off-diagonal entries of the intervention specificity matrix are small relative to diagonals. If no, stop and write C1 as a negative result.
- **G2 (persistence).** Selectivity ratio > threshold at horizon 10, and edited rollouts survive autoregressive re-encoding better than matched-random baselines. If no, stop and write up as a finding about autoregressive correction in diffusion WMs.
- **G3 (transfer).** Targeted edits yield more real-environment failures than matched-random edits at equivalent compute. If no, stop.

## Limitations

See `LIMITATIONS.md`.

## Archived v1

See `archive/v1/`. The earlier perturbation-preset and adaptive-curriculum work established that naive uniform perturbation of DIAMOND's sampling loop collapses on Pong and that scheduling mitigates it. This repo pursues the deeper question those results raised: are the perturbations causally meaningful in the first place?
