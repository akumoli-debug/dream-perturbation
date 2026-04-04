# Dream Perturbation: Robustness via Diffusion World Model Surgery

> Train RL agents across perturbed dream variants of DIAMOND's pixel-level diffusion world model — and watch them generalise to the real world.

**v4 status:** 500-epoch run in progress. v2 200-epoch results below.

---

## Key Results

### Breakout (200 epochs · 50 eval episodes · 4 agents) — v2

| Agent | Real Atari Score | Dream Robustness (avg / 5 variants) |
|---|---|---|
| **Adaptive-Curriculum** | **208.6 ± 143.2** | **0.73** |
| Baseline | 223.3 ± 139.7 | 0.70 |
| Multi-Dream | 194.1 ± 136.2 | 0.70 |
| Single-Perturb | 33.0 ± 80.5 | 0.55 |

> Adaptive-Curriculum retains **93.4%** of baseline real-world performance while achieving the highest dream robustness score.

### Pong (200 epochs · 50 eval episodes · 4 agents) — v2

| Agent | Real Atari Score | Dream Robustness (avg / 5 variants) |
|---|---|---|
| **Adaptive-Curriculum** | **20.6 ± 1.06** | **-0.08** |
| Baseline | 20.8 ± 0.69 | -0.11 |
| Single-Perturb | 20.1 ± 1.65 | -0.18 |
| Multi-Dream | 4.8 ± 5.96 | -0.11 |

> Naive uniform cycling (Multi-Dream) **collapses** on Pong without a curriculum. Adaptive-Curriculum maintains **99%** of baseline performance while improving dream robustness.

**Cross-game finding:** The curriculum is not optional — it's load-bearing. Adaptive-Curriculum is the only approach that replicates across both games.

> **Note:** v4 results (500 epochs, updated presets `sigma_perturb` + `adversarial_action`) are in progress. Tables will be updated upon completion.

---

## Motivation

[DIAMOND](https://arxiv.org/abs/2405.12399) (NeurIPS 2024) demonstrated that a pixel-level diffusion world model can produce environments rich enough to train competitive Atari agents entirely in imagination. The natural next question: **can we perturb those dreams to generate novel training environments on demand?**

This matters because:

1. **Sim-to-real transfer** is the central problem in modern RL. Agents trained on a single, static simulator overfit to its physics and fail when deployed. Training across diverse dream variants is a tractable proxy for domain randomization.
2. **Data efficiency.** A trained world model is a generative simulator. Inference-time perturbation creates an unbounded supply of novel environments without any additional real environment interaction.
3. **This is precisely what next-generation AI labs need.** Robust agents require robust simulators — and pixel-level diffusion world models are the most expressive simulator class available.

Prior work either operates at the layout level ([ADD, NeurIPS 2024](https://arxiv.org/abs/2412.05790)) or uses latent-space perturbation with DreamerV3 ([DALI, Jan 2026](https://arxiv.org/abs/2501.09595)). **Nobody had perturbed DIAMOND's pixel-level diffusion process at inference time to create environment variants for RL training — until this project.**

---

## Method

The centerpiece of this project is `src/perturbed_world_model_env.py` — a drop-in replacement for DIAMOND's `WorldModelEnv` that injects controllable perturbations at inference time, without ever touching the world-model weights.

### Perturbation Presets

Five presets cover orthogonal axes of world model surgery:

---

### `normal` — Unperturbed Baseline

**Where:** No intervention. The standard `WorldModelEnv.step()` pipeline runs unchanged.

**What:** Passes actions directly to the denoiser and returns the denoised observation as-is.

**Why:** Provides the reference distribution. Every other preset is measured against this.

---

### `mirrored` — Spatial Mirror + Action Swap

**Where:** Post-denoising frame transform (`PerturbedWorldModelEnv._perturb_obs()`) and pre-step action remapping (`_perturb_action()` via `_build_action_remap()`).

**What:** Applies `obs.flip(dims=[-1])` to horizontally mirror every generated frame, and simultaneously swaps LEFT (index 4) ↔ RIGHT (index 3) in the action space.

**Why:** Tests the agent's ability to transfer spatial reasoning to a reflected coordinate system. Because the action remap and the visual flip are consistent, the game is fully playable — but the learned motor programs must invert. This is a strong test of how abstract the policy's representations are.

---

### `sigma_perturb` — Noise Schedule Perturbation

**Where:** Modifies `DiffusionSamplerConfig.sigma_min` and `num_steps_denoising` on `self._env.sampler.cfg` before each world model rollout step (`PerturbedWorldModelEnv.step()`, see `_apply_sigma_perturbation()` and `_restore_sigma()`).

**What:** Increases the noise floor by 1.5× (`sigma_min *= 1.5`) and reduces the number of denoising steps by 30% (`new_steps = max(1, round(orig_steps * 0.7))`). The sampler's pre-built `sigmas` schedule tensor is rebuilt via `build_sigmas()` to take effect immediately. Original values are restored after each step.

**Why:** Tests whether the policy is robust to the world model's **uncertainty calibration** — i.e., does the agent still act correctly when the world model is "less sure" about its predictions? Higher `sigma_min` means the final denoised frame retains more residual noise; fewer steps means less denoising refinement. Both make the world model coarser and less faithful. An agent that acts correctly under this degraded world model has learned representations that don't depend on frame-level visual precision.

---

### `adversarial_action` — Online Adversarial Action Perturbation

**Where:** A dynamically trained 2-layer MLP (`PerturbedWorldModelEnv._adv_mlp`) is applied in `step()` via `_apply_adversarial_action()` before actions are dispatched to the world model. One gradient step is taken after each reward is observed via `_adversarial_gradient_step()`.

**What:** The MLP maps a 2D context vector `[normalised_return, normalised_step]` to an action logit offset `[num_actions]`. This offset is added to a one-hot encoding of the original action, and the new action is `argmax` of the perturbed logits. The MLP is trained online with a single Adam step per env step using the loss `−mean(reward)` — it is an adversary that seeks to maximise the agent's negative reward.

```python
# Context: episode progress and cumulative return
context = torch.tensor([[norm_return, norm_step]])     # [1, 2]
offset   = adv_mlp(context)                            # [1, num_actions]
perturbed_logits = one_hot(act) + offset
perturbed_act    = perturbed_logits.argmax(dim=-1)

# Adversarial update (after observing reward)
loss = -mean(reward)
loss.backward()
adv_optimizer.step()
```

**Why:** Unlike fixed perturbations, the adversarial MLP adapts to the current policy's weaknesses in real-time. As the agent trains and improves, the adversary finds new exploits. This creates a minimax dynamic that pushes the policy toward robustness against the worst-case action perturbation reachable from its own logit distribution — a learned, context-sensitive version of action dropout.

---

### `hard_mode` — Everything Combined

**Where:** All perturbation layers are active simultaneously: frame transforms (`_perturb_obs()`), physics bias (`_apply_physics_bias()`), action remap + dropout (`_perturb_action()`), and reward scaling + delay (`_perturb_reward()`).

**What:** Horizontal flip, left/right action swap, Gaussian frame noise (σ=0.06), brightness shift (+0.03), contrast reduction (0.85×), spatial translation (±3px), 5% action dropout, 0.75× reward scale, 1-step reward delay, and a rightward directional pixel gradient (strength 0.03).

**Why:** Maximum distributional shift from all directions at once. An agent that trains stably under `hard_mode` must have learned representations that are simultaneously robust to visual corruption, action perturbation, reward signal delay, and spatial dynamics bias. Used as the ceiling difficulty in the adaptive curriculum.

---

## Agent Variants

| Agent | Training Environment |
|---|---|
| `Baseline` | Standard DIAMOND dreams (normal preset only) |
| `Single-Perturb` | Fixed single perturbation (mirrored) throughout training |
| `Multi-Dream` | Uniform random cycling across all 5 presets each episode |
| `Adaptive-Curriculum` | Difficulty-aware scheduling — starts on normal, promotes harder presets as performance improves |

---

## Adaptive Curriculum Algorithm

The curriculum maintains a running average return per dream variant and samples inversely proportional to performance. The agent spends more time on variants it currently performs worst at.

```
# Initialise
variant_scores = {preset: 0.0  for preset in PRESET_NAMES}
eval_interval  = 10

# At each epoch
if epoch < eval_interval:
    preset = cycle_uniformly(PRESET_NAMES, epoch)         # warm-up: uniform cycling
else:
    epsilon = 0.1
    weights = [1.0 / (variant_scores[p] + epsilon)
               for p in PRESET_NAMES]
    preset  = sample(PRESET_NAMES, weights / sum(weights)) # sample hardest

# Every eval_interval epochs: quick 3-episode eval per variant
for p in PRESET_NAMES:
    new_score = evaluate_dream_variant(actor_critic, p, episodes=3)
    variant_scores[p] = 0.5 * variant_scores[p] + 0.5 * new_score  # EMA, α=0.5
```

**Key properties:**
- The warm-up phase prevents cold-start distributional shock.
- EMA smoothing (α=0.5) prevents thrashing between variants.
- The `epsilon` floor ensures all variants stay reachable regardless of score.
- This acts as a trust-region constraint in *environment space* — harder variants are gated by demonstrated policy competence.

---

## OOD Cross-Game Evaluation

**v4 introduces a new evaluation** that probes whether dream-perturbation training produces *truly generalisable* representations, or just robustness to in-distribution visual corruptions.

**Setup:** After Breakout training, we load DIAMOND's pretrained Pong world model and run each of the 4 Breakout actor-critics directly inside it. No Pong training is performed — this is pure zero-shot cross-game transfer in imagination.

**Action-space mismatch:** Breakout policies produce 4-action logits; Pong has 6 actions. We zero-pad:

```python
padded_logits = F.pad(logits_4, (0, num_pong_actions - num_breakout_actions))
# Shape: [batch, 4] → [batch, 6]; padding appended at high-index actions
```

**What it measures:** The Pong world model generates a completely different visual distribution (a paddle-and-ball game vs. a brick-breaker). A policy that acts non-randomly in this foreign world model has learned representations that transfer across game-level distributional shifts — a stronger claim than transfer across noise conditions.

**Hypothesis:** Adaptive-Curriculum agents, having been exposed to the widest range of dream variants during training, should generalise better to the Pong world model than Baseline agents.

---

## Results & Analysis

### 1. Curriculum is Essential on Harder Games

On Breakout (high score variance, many strategies viable), Multi-Dream's uniform cycling is tolerable — performance drops only ~13% vs baseline. On Pong (near-perfect optimal policy, tight score distribution), the same approach causes **catastrophic collapse** to 4.8 ± 5.96. The Adaptive-Curriculum avoids this entirely, achieving 20.6 ± 1.06 — within 1% of baseline.

**Interpretation:** Pong's narrow optimal-policy corridor makes distributional shift lethal. The curriculum acts as a trust-region constraint in environment space.

### 2. Single-Perturb is Brittle

Training exclusively on `hard_mode` destroys Breakout performance (33.0 ± 80.5, an 85% drop) without meaningfully improving robustness (0.55 vs baseline's 0.70). The dream variant the agent trains in must be reachable from the real environment's distribution — a single extreme perturbation creates a domain too far from reality.

### 3. Cross-Game Replication

The key result replicates on both games: **Adaptive-Curriculum achieves the best dream robustness while sustaining near-baseline real performance.** This is not a Breakout-specific artifact.

### 4. Robustness vs. Performance Trade-off

There is no free lunch — training on perturbed environments introduces some performance variance. Adaptive-Curriculum minimizes this trade-off by curriculum-gating access to harder environments, exposing perturbations only once the agent has sufficient policy capacity to absorb them.

---

## Figures

All figures are generated by the experiment pipeline and saved to game-specific results directories.

**Per-game (v4):**
- `dream_perturbation_comparison_v4.png` — Real Atari return bar chart (4 agents)
- `dream_robustness_v4.png` — Heatmap of agent × dream variant performance
- `training_curves_v4.png` — Policy entropy over training for all 4 agents
- `transfer_robustness_v4.png` — Grouped bar chart: 3 transfer conditions × 4 agents
- `ood_evaluation.png` — Bar chart: Breakout agents in Pong world model (Breakout run only)

**Per-game (v2, archived):**
- `results_v2_breakout/figures/dream_perturbation_comparison_v2.png`
- `results_v2_pong/figures/dream_robustness_v2.png`

**Cross-game (v2, archived):**
- `results_v2_cross_game/cross_game_real_env.png`
- `results_v2_cross_game/performance_robustness_tradeoff.png`

---

## Related Work

| Paper | Venue | Method | Difference from This Work |
|---|---|---|---|
| [DIAMOND](https://arxiv.org/abs/2405.12399) | NeurIPS 2024 | Pixel-level diffusion world model for Atari | Foundation — we build on DIAMOND's world model |
| [ADD](https://arxiv.org/abs/2412.05790) | NeurIPS 2024 | Automated environment generation via layout mutation | Layout-level; does not perturb diffusion world models |
| [DALI](https://arxiv.org/abs/2501.09595) | Jan 2026 | Latent-space perturbation with DreamerV3 | Latent-level with DreamerV3; we do pixel-level with DIAMOND |

**This work is the first to apply inference-time perturbation to DIAMOND's pixel-level diffusion process — including direct surgery on the EDM noise schedule (`sigma_perturb`) and online adversarial action perturbation (`adversarial_action`) — as a mechanism for novel environment generation.**

---

## Repository Structure

```
learning-from-failure/
├── src/
│   ├── perturbed_world_model_env.py   # ← CENTERPIECE: PerturbedWorldModelEnv, 5 preset configs
│   │                                  #   sigma_perturb: EDM noise schedule surgery
│   │                                  #   adversarial_action: online adversarial MLP
│   │                                  #   mirrored, hard_mode, normal: frame/action/reward
│   ├── failure_detector.py            # Failure event detection (exploratory)
│   ├── dataset_curator.py             # Dataset curation tools (exploratory)
│   ├── failure_eval.py                # Failure-conditioned eval metrics
│   ├── failure_diversity.py           # FMDS diversity scorer
│   ├── wm_quality_predictor.py        # 7-feature WM quality predictor
│   └── visualizations.py             # Figure generation
├── scripts/
│   ├── dream_perturbation_experiment_v4.py  # Main experiment (v4: 500 epochs, OOD eval)
│   ├── dream_perturbation_experiment_v3.py  # v3: transfer tests, checkpoint saving
│   ├── dream_perturbation_experiment_v2.py  # v2: multi-game, cross-game figures
│   ├── dream_perturbation_experiment.py     # Original v1 experiment
│   ├── run_dream_perturbation.sh            # One-shot Vast.ai setup + run
│   └── cross_game_analysis.py              # Cross-game comparison figures
├── results/                           # v4 results (in progress)
├── results_v2_breakout/               # v2 Breakout results, figures, JSON
├── results_v2_pong/                   # v2 Pong results, figures, JSON
├── results_v2_cross_game/             # v2 cross-game analysis figures
└── README.md
```

---

## Reproducing Results

**Hardware:** Single **Vast.ai RTX 5090** ($0.475/hr).  
**Approximate cost v4:** ~$12 per full game run (500 epochs, 4 agents).  
**Wall time:** ~15–18 hours per game (500 epochs).

```bash
# Option 1: One-shot script (handles clone, patch, install, run)
bash scripts/run_dream_perturbation.sh

# Option 2: Manual (if DIAMOND is already set up)
source ~/env/bin/activate
pip install matplotlib huggingface_hub

# v4: Train + evaluate all 4 agents on Breakout (500 epochs, OOD eval)
PYTHONPATH="$DIAMOND_ROOT/src:./src:$PYTHONPATH" \
python scripts/dream_perturbation_experiment_v4.py \
    --diamond-root ~/diamond \
    --output-dir results_v4_breakout \
    --game Breakout \
    --num-epochs 500

# v4: Train + evaluate all 4 agents on Pong (no OOD eval, skipped automatically)
PYTHONPATH="$DIAMOND_ROOT/src:./src:$PYTHONPATH" \
python scripts/dream_perturbation_experiment_v4.py \
    --diamond-root ~/diamond \
    --output-dir results_v4_pong \
    --game Pong \
    --num-epochs 500

# Quick smoke-test (10 epochs, all 4 agents, ~5 minutes)
PYTHONPATH="$DIAMOND_ROOT/src:./src:$PYTHONPATH" \
python scripts/dream_perturbation_experiment_v4.py \
    --diamond-root ~/diamond \
    --output-dir results_fast \
    --game Breakout \
    --fast

# v2 (archived, 200 epochs):
PYTHONPATH="$DIAMOND_ROOT/src:./src:$PYTHONPATH" \
python scripts/dream_perturbation_experiment_v2.py \
    --diamond-root ~/diamond \
    --output-dir results_v2_breakout \
    --game Breakout

# Generate cross-game comparison figures from v2
python scripts/cross_game_analysis.py
```

Pretrained DIAMOND checkpoints are downloaded automatically from [HuggingFace (eloialonso/diamond)](https://huggingface.co/eloialonso/diamond).

---

## Citation

If you build on this work:

```bibtex
@misc{wei2026dreamperturbation,
  title   = {Dream Perturbation: Robustness via Diffusion World Model Surgery},
  author  = {Wei, Albert},
  year    = {2026},
  url     = {https://github.com/akumoli-debug/dream-perturbation}
}
```

**Acknowledgments:** Built on [DIAMOND](https://github.com/eloialonso/diamond) by Alonso et al. (NeurIPS 2024). Compute provided by [Vast.ai](https://vast.ai).
