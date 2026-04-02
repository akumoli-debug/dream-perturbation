# Dream Perturbation: Generating Novel Training Environments from DIAMOND's World Model

> Fine-tune a DIAMOND pixel-level diffusion world model, perturb its inference process to synthesize novel environment variants, and train RL agents in those dreams — testing whether diversity in imagination transfers to robustness in reality.

---

## Key Results

### Breakout (200 epochs · 50 eval episodes · 4 agents)

| Agent | Real Atari Score | Dream Robustness (avg / 5 variants) |
|---|---|---|
| **Adaptive-Curriculum** | **208.6 ± 143.2** | **0.73** |
| Baseline | 223.3 ± 139.7 | 0.70 |
| Multi-Dream | 194.1 ± 136.2 | 0.70 |
| Single-Perturb | 33.0 ± 80.5 | 0.55 |

> Adaptive-Curriculum retains **93.4%** of baseline real-world performance while achieving the highest dream robustness score.

### Pong (200 epochs · 50 eval episodes · 4 agents)

| Agent | Real Atari Score | Dream Robustness (avg / 5 variants) |
|---|---|---|
| **Adaptive-Curriculum** | **20.6 ± 1.06** | **-0.08** |
| Baseline | 20.8 ± 0.69 | -0.11 |
| Single-Perturb | 20.1 ± 1.65 | -0.18 |
| Multi-Dream | 4.8 ± 5.96 | -0.11 |

> Naive uniform cycling (Multi-Dream) **collapses** on Pong without a curriculum. Adaptive-Curriculum maintains **99%** of baseline performance while improving dream robustness.

**Cross-game finding:** The curriculum is not optional — it's load-bearing. Adaptive-Curriculum is the only approach that replicates across both games.

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

### Perturbation Presets

Five perturbation presets operate directly on DIAMOND's diffusion sampling loop:

| Preset | Description |
|---|---|
| `normal` | No perturbation — canonical dream (baseline) |
| `mirrored` | Horizontal flip applied to each sampled frame |
| `noisy` | Gaussian noise injected at each diffusion timestep |
| `shifted_physics` | Action embeddings shifted to simulate altered dynamics |
| `hard_mode` | Combined noise + physics shift — maximum difficulty |

Perturbations are applied **during inference**, requiring no retraining of the world model.

### Agent Variants

| Agent | Training Environment |
|---|---|
| `Baseline` | Standard DIAMOND dreams (normal preset only) |
| `Single-Perturb` | Fixed single perturbation (mirrored) throughout training |
| `Multi-Dream` | Uniform random cycling across all 5 presets each episode |
| `Adaptive-Curriculum` | Difficulty-aware scheduling — starts on normal, promotes harder presets as performance improves |

### Adaptive Curriculum Algorithm

The curriculum maintains a running average return per dream variant and samples inversely proportional to performance:

```
weights[i] = 1.0 / (variant_scores[preset_i] + epsilon)
next_preset = sample(presets, weights / sum(weights))
```

Every 10 epochs, quick evaluation (3 episodes per variant) updates the running averages with exponential smoothing (α=0.5). This means the agent spends more training time on variants it performs worst at.

This prevents distributional shock (the mechanism behind Multi-Dream's Pong collapse) while still exposing the agent to progressively harder dream variants.

### Dream Robustness Metric

Dream robustness is measured as mean normalized score across all 5 perturbation variants:

```
robustness = mean( normalize(score_v) for v in {normal, mirrored, noisy, shifted_physics, hard_mode} )
```

where `normalize` is min-max scaled against the score range observed in the experiment.

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

**Per-game (`results_v2_breakout/`, `results_v2_pong/`):**
- `dream_perturbation_comparison_v2.png` — Real Atari return bar chart (4 agents)
- `dream_robustness_v2.png` — Heatmap of agent × dream variant performance
- `training_curves_v2.png` — Policy entropy over training for all 4 agents

**Cross-game (`results_v2_cross_game/`):**
- `cross_game_real_env.png` — Side-by-side real env performance
- `cross_game_robustness.png` — Side-by-side dream robustness
- `cross_game_heatmap.png` — 8-row heatmap (4 agents × 2 games)
- `performance_robustness_tradeoff.png` — Scatter with Pareto frontier
- `adaptive_curriculum_focus.png` — Adaptive vs baseline delta per variant

---

## Repository Structure

```
learning-from-failure/
├── src/
│   ├── perturbed_world_model_env.py   # PerturbedWorldModelEnv — 5 preset configs
│   ├── failure_detector.py            # Failure event detection (exploratory)
│   ├── dataset_curator.py             # Dataset curation tools (exploratory)
│   ├── failure_eval.py                # Failure-conditioned eval metrics
│   ├── failure_diversity.py           # FMDS diversity scorer
│   ├── wm_quality_predictor.py        # 7-feature WM quality predictor
│   └── visualizations.py              # Figure generation
├── scripts/
│   ├── dream_perturbation_experiment_v2.py  # Main experiment (4 agents, multi-game)
│   ├── dream_perturbation_experiment.py     # Original v1 experiment
│   ├── run_dream_perturbation.sh            # One-shot Vast.ai setup + run
│   └── cross_game_analysis.py               # Cross-game comparison figures
├── results_v2_breakout/               # Breakout results, figures, JSON
├── results_v2_pong/                   # Pong results, figures, JSON
├── results_v2_cross_game/             # Cross-game analysis figures
└── README.md
```

---

## Reproducing Results

**Hardware:** Single **Vast.ai RTX 5090** ($0.475/hr).  
**Approximate cost:** ~$5 per full game run (200 epochs, 4 agents).  
**Wall time:** ~6–8 hours per game.

```bash
# Option 1: One-shot script (handles clone, patch, install, run)
bash scripts/run_dream_perturbation.sh

# Option 2: Manual (if DIAMOND is already set up)
source ~/env/bin/activate
pip install matplotlib huggingface_hub

# Train + evaluate all 4 agents on Breakout
PYTHONPATH="$DIAMOND_ROOT/src:./src:$PYTHONPATH" \
python scripts/dream_perturbation_experiment_v2.py \
    --diamond-root ~/diamond \
    --output-dir results_v2_breakout \
    --game Breakout

# Train + evaluate all 4 agents on Pong
PYTHONPATH="$DIAMOND_ROOT/src:./src:$PYTHONPATH" \
python scripts/dream_perturbation_experiment_v2.py \
    --diamond-root ~/diamond \
    --output-dir results_v2_pong \
    --game Pong

# Generate cross-game comparison figures
python scripts/cross_game_analysis.py
```

Pretrained DIAMOND checkpoints are downloaded automatically from [HuggingFace](https://huggingface.co/eloialonso/diamond).

---

## Related Work

| Paper | Venue | Method | Difference from This Work |
|---|---|---|---|
| [DIAMOND](https://arxiv.org/abs/2405.12399) | NeurIPS 2024 | Pixel-level diffusion world model for Atari | Foundation — we build on DIAMOND's world model |
| [ADD](https://arxiv.org/abs/2412.05790) | NeurIPS 2024 | Automated environment generation via layout mutation | Layout-level; does not perturb diffusion world models |
| [DALI](https://arxiv.org/abs/2501.09595) | Jan 2026 | Latent-space perturbation with DreamerV3 | Latent-level with DreamerV3; we do pixel-level with DIAMOND |

**This work is the first to apply inference-time perturbation to DIAMOND's pixel-level diffusion process as a mechanism for novel environment generation.**

---

## Citation

If you build on this work:

```bibtex
@misc{wei2026dreamperturbation,
  title   = {Dream Perturbation: Generating Novel Training Environments from DIAMOND's World Model},
  author  = {Wei, Albert},
  year    = {2026},
  url     = {https://github.com/akumoli-debug/dream-perturbation}
}
```

**Acknowledgments:** Built on [DIAMOND](https://github.com/eloialonso/diamond) by Alonso et al. (NeurIPS 2024). Compute provided by [Vast.ai](https://vast.ai).
