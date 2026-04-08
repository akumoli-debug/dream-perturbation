# dream-perturbation

Mechanistic interpretability experiments on [DIAMOND](https://github.com/eloialonso/diamond), a diffusion-based world model. Probes DIAMOND's U-Net to understand where and how spatial information is encoded, whether that encoding is causally functional, and how it compares across games and architectures.

---

## Findings

### 1. Spatial encoding is readable with linear probes (R²=0.625)
Linear probes trained on ALE RAM ball_y labels achieve **R²=0.6246 ± 0.0021** at `u_blocks.0.resblocks.1` in Pong, and **R²=0.3806 ± 0.0138** in Breakout (5-seed, 95% CI). The encoding profile peaks at the bottleneck/early decoder — near zero at input and output layers. Higher R² in Pong is consistent with ball trajectory being more globally predictive in Pong's dynamics than Breakout's.

### 2. Spatial encoding is causally functional — dose-response confirmed
Activation patching with mixing weight α ∈ {0, 0.25, 0.5, 0.75, 1.0} produces a **monotonic dose-response curve** (p=0.0135 at α=1.0, n=500 pairs). The best layer `u_blocks.0.resblocks.1` shows Cohen's d=+0.256 (p<0.001) in the multi-layer sweep. Effect is small — consistent with strong manifold correction by subsequent denoising steps.

### 3. Perturbations that destroy spatial encoding

| Perturbation | Probe R² | Status |
|---|---|---|
| Baseline | 0.303 | Intact |
| Action mask (all NOOP) | 0.303 | Intact — actions don't drive spatial encoding |
| Drop 1/4 context frames | 0.303 | Intact |
| Drop 2/4 context frames | 0.258 | Mild degradation |
| Obs noise σ=0.05 | 0.251 | Mild degradation |
| Drop 3/4 context frames | 0.005 | **Encoding collapsed** |
| Obs noise σ=0.20 | −0.020 | **Encoding collapsed** |

### 4. Architecture comparison: DIAMOND vs VAE

| Architecture | Best layer | Peak R² | Location |
|---|---|---|---|
| DIAMOND (diffusion U-Net) | `u_blocks.0.resblocks.1` | **0.618** | Bottleneck/early decoder |
| VAE (conv encoder-decoder) | `conv_8` | **0.334** | Deepest encoder |

DIAMOND encodes spatial information more strongly and in a different location than a VAE. The diffusion denoising process builds spatial representations in the decoder pathway; VAE concentrates them at the encoder bottleneck.

### 5. Temporal sensitivity cliff

| Frames dropped | Mean\|Δframe\| | Multiplier |
|---|---|---|
| 0 (baseline) | 0.002 | 1× |
| 1 of 4 | 0.047 | **21×** |
| 2 of 4 | 0.175 | **80×** |
| 3 of 4 | 0.224 | **102×** |

### 6. Cross-game causal efficiency
Dose-response on Pong (R²=0.625) vs Breakout (R²=0.381):
- Pong: Cohen's d=+0.163, p=0.031 — significant causal signal
- Breakout: Cohen's d=−0.016, p=0.572 — no significant signal
- Higher probe R² correlates with stronger causal influence across games

---

## Setup

```bash
git clone https://github.com/eloialonso/diamond ~/diamond
git clone https://github.com/akumoli-debug/dream-perturbation
cd dream-perturbation
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install gymnasium[atari] ale-py autorom matplotlib huggingface_hub opencv-python-headless omegaconf hydra-core wandb einops
AutoROM --accept-license
huggingface-cli download eloialonso/diamond --local-dir ~/diamond/pretrained
export DIAMOND_ROOT=~/diamond
export PYTHONPATH="$DIAMOND_ROOT/src:./src:$PYTHONPATH"
```

Requires PyTorch with CUDA 12.8 (`--index-url https://download.pytorch.org/whl/cu128`) for RTX 5090. Single GPU sufficient.

---

## Running the experiments

```bash
# Collect consecutive 4-frame context with RAM labels
python3 src/fix1_consecutive_frames.py

# Train linear probes (5 seeds, 95% CI)
python3 src/fix2_train_probes_v2.py --game Pong
python3 src/fix2_train_probes_v2.py --game Breakout

# Also needed for individual .pt probe files (used by causal experiments)
python3 src/step2_collect_ram_labels.py
python3 src/step4_train_probes.py --game Pong

# Causal dose-response curve
python3 src/add1_dose_response.py --game Pong --n_pairs 500

# Probe R² under perturbation
python3 src/add2_probe_predicts_return.py --game Pong

# Architecture comparison (trains VAE from scratch, ~60 min)
python3 src/add3_second_architecture.py --game Pong

# Multi-layer causal ablation with probe-guided measurement
python3 src/expA_multilayer_causal.py --game Pong --n_pairs 200

# Cross-game dose-response + causal efficiency
python3 src/expB_cross_game_efficiency.py --n_pairs 250
```

---

## Results



---

## Source files

| File | Purpose |
|---|---|
| `src/fix1_consecutive_frames.py` | Collect 4-frame consecutive context + RAM labels |
| `src/fix2_train_probes_v2.py` | Probe training, 5 seeds, 95% CI |
| `src/fix3_causal_intervention.py` | Binary activation patch experiment |
| `src/add1_dose_response.py` | Causal dose-response curve (α sweep) |
| `src/add2_probe_predicts_return.py` | Probe R² under perturbation |
| `src/add3_second_architecture.py` | VAE world model + probe comparison |
| `src/expA_multilayer_causal.py` | Multi-layer causal profile with Cohen's d |
| `src/expB_cross_game_efficiency.py` | Cross-game dose-response + efficiency ratio |
| `src/step3_hook_activations.py` | U-Net forward hook system |
| `src/step4b_orthogonalize.py` | Gram-Schmidt confound removal |
| `src/step5_geometric_steering_env.py` | Activation steering env |
| `src/finetune_conditioned_diamond.py` | ball_y conditioning fine-tune |
| `src/perturbed_world_model_env.py` | DIAMOND perturbation wrapper |
| `src/wm_quality_predictor.py` | World model quality proxy features |
