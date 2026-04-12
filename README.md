# Dream Perturbation

**Thesis.** Diffusion world models may contain editable internal directions corresponding to environment state. We test whether interventions on these directions generate selective, temporally coherent counterfactual rollouts, and whether those rollouts provide more useful training signal than untargeted perturbations.

## Claim hierarchy

- **C1** — Internal directions in DIAMOND's U-Net encode environment variables in an intervention-relevant way.
- **C2** — Interventions on those directions produce selective and temporally coherent counterfactual rollouts.
- **C3** — Targeted rollouts improve downstream failure discovery over matched-magnitude random perturbation, and those failures transfer to the real Atari environment.

Each claim is independently publishable. If C3 fails, C1+C2 stand as a mechanistic interpretability result. If C2 fails, C1 stands as a probing result. If C1 fails, the negative result is itself a finding about diffusion world model internals.

## Experiments

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
