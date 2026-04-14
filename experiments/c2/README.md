# C2 Pilot: Preregistration

**Status:** preregistered, no intervention code written yet.
**Branch:** `experiments/c2-pilot`.
**Depends on:** C1 (probe sweep), reproduced on this branch — `data/probe_results.csv` shows s3_mid/pooled/sigma[2]/ball_y R2 = 0.7147; refit on freshly-built dataset yields 0.7169 (see `src/steering/refit_c2_probe.py`).

---

## Claim

An intervention that adds `k * sigma_norm * direction` to the pooled-channel component of DIAMOND's U-Net activations at site `s3_mid` at denoising step sigma[2] — where `direction` is the L2-normalized C1 ridge-probe coefficient vector for `ball_y` — produces rollout trajectories whose measured `ball_y` shifts monotonically with `k`, while leaving non-target variables (`paddle_y`) substantially less affected, and this effect is not reproduced by matched-L2 random-direction perturbations nor by the same intervention applied to an untrained DIAMOND.

## Intervention spec

- **Site:** `s3_mid`, pooled representation (spatial mean across 8x8 -> 64-d vector).
- **Denoising timestep:** sigma_idx = 2 (out of 3 sigmas collected). No intervention at sigma_idx 0 or 1.
- **Direction:** L2-normalized coefficient of the C1 ridge probe saved at `data/steering_vectors/c2_pilot_s3mid_sigma2_bally.pt`.
- **Scale:** `k * sigma_norm` where sigma_norm = L2 norm of unnormalized probe coef (= 487.43 for the committed vector).
- **Injection mechanism:** forward hook on the `s3_mid` module. Hook pools activations to channel space, adds `k * sigma_norm * direction` broadcast across spatial dims, reinjects. Fires only at sigma[2] denoising step.
- **Hook sanity test:** with k=0, forward pass output must be bit-identical to unedited pass. Required before any non-zero k is run.

## Pilot protocol

- **N conditioning windows:** 50 (from unseen held-out split of the Pong dataset, temporally after C1 training split).
- **Rollout length:** 10 frames per window.
- **k values:** {-2, -1, 0, +1, +2} (units of sigma_norm). k=0 is the unedited control.
- **Arms:**
  - A. **Steered (trained DIAMOND):** probe-direction intervention, 4 non-zero k values x 50 windows = 200 rollouts.
  - B. **Unedited (trained DIAMOND):** 50 rollouts.
  - C. **Matched-random (trained DIAMOND):** random unit direction sampled once per rollout, same k magnitudes, 4 x 50 = 200 rollouts.
  - D. **Untrained-model control:** same probe-direction intervention, k=+2 only, 50 rollouts.
- **Labeler:** rerun `read_ram_labels` on simulated frames via a pixel-based ball detector (RAM is not available in dreamed rollouts). Detector spec and validation protocol documented separately in `experiments/c2/labeler.md` when written.

## Gates (pass/fail, committed in advance)

**Target variable:** `ball_y`.
**Non-target panel (primary):** `{left_paddle_y, right_paddle_y}`. Paddles are agent-controlled, not physics-coupled to ball position, so they remain valid distractors over multi-step rollouts. `ball_x` and `ball_vx` are excluded because they become mechanistically coupled to `ball_y` through bounce dynamics by rollout step ~5.
**Sentinel non-target (reported separately, not in panel):** `right_paddle_y` alone.

- **G1a (screening).** Probe R2 for target at intervention site >= 0.5. Already satisfied: s3_mid/pooled/sigma[2]/ball_y R2 = 0.7169. This is eligibility to run C2, not success.

- **G1b (target responsiveness, intervention gate).** In arm A, Pearson r between k and per-rollout mean Delta-ball_y (edited minus unedited for same conditioning window) >= 0.5, with 95% bootstrap CI lower bound > 0. In arm C (matched-random), |r| <= 0.15. r and its CI are reported continuously regardless of pass/fail; G1b is the minimum intervention-movement threshold, not a claim about representation strength.

- **G2-immediate (selectivity, step 1).** At rollout step 1, k=+2:
    selectivity_ratio = |mean Delta-ball_y| / (mean_over_panel(|mean Delta-non_target|) + epsilon)
  Primary gate: selectivity_ratio >= 3.0 with 95% bootstrap CI lower bound >= 2.0.

- **G2-sustained (selectivity, horizon 10).** Same ratio with Delta values averaged across rollout steps 1-10. Primary gate: selectivity_ratio >= 3.0 with bootstrap CI lower bound >= 2.0. This is the load-bearing claim. If immediate selectivity holds but sustained selectivity collapses, that is a finding about diffusion error-correction attenuating targeted edits, not a causal steering success.

- **G2-sentinel (reported, not gating).** Same ratio computed with `right_paddle_y` alone as denominator. Reported separately so the spatially-nearby-but-weakly-decoded variable is visible; if it diverges from the panel average, note it.

- **G3 (untrained-model control).** In arm D, Pearson r between k and Delta-ball_y <= 0.2. If r > 0.5, pilot is treated as confounded (editing architecture, not learned structure) and the steered-arm result is not claimed as causal evidence regardless of G1b/G2 outcomes.

- **H2 (secondary, pre-registered, not gating).** If ball_vx steering is also run, predict it fails G1b (r < 0.3) even at matched L2 magnitude. This is a test of the representation-gap hypothesis that weakly-encoded variables cannot be causally steered via probe-extracted directions.

## What does NOT count as success

- G1 passes but G3 also shows r > 0.5 -> architecture effect, not causal claim.
- G1 passes but G2 < 3.0 -> probe direction is not a selective causal handle (decodes ball_y but co-moves paddle_y). This is the causal-vs-decoding divergence flagged in LIMITATIONS.md.
- G1 pass only on a subset of the 50 windows with post-hoc filtering. No post-hoc subset selection is permitted before verdict is written.

## Scale-up trigger

Full C2 (N=500, horizon up to 20, autoregressive re-encoding stability tests) proceeds only if G1 AND G2 AND G3 all pass at pilot N=50. If any fails, the finding is the failure mode itself; no scale-up.

## Artifacts required at pilot completion

- `results_v3/c2_pilot/verdict.md` — pass/fail on each gate with bootstrap CIs, written before scale-up decision.
- `results_v3/c2_pilot/artifact.gif` — one side-by-side k=+2 edited vs unedited rollout from the same conditioning window.
- Raw per-rollout data (parquet or jsonl) for arms A, B, C, D.
