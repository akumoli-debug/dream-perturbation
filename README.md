# dream-perturbation

Inference-time interventions on DIAMOND, a diffusion world model, to probe and perturb its learned simulator dynamics. Built on [DIAMOND](https://github.com/eloialonso/diamond) (Alonso et al., 2024).

## What this repo does

Three experiments on DIAMOND's Atari Pong/Breakout world model:

### 1. Spatial encoding probes
Linear probes trained on ALE RAM ground-truth ball positions, applied to every ResBlock in DIAMOND's U-Net. Finds which layers linearly encode ball position.

**Result:** `u_blocks.0.resblocks.1` achieves R²=0.548 for ball_y in Pong. Breakout R²=0.05 — encoding is game-dependent and proportional to causal relevance.

### 2. Geometric steering (Path C)
Attempts to steer ball position at inference time by injecting the probe's weight vector into U-Net activations during denoising, and via fine-tuning with explicit ball_y conditioning.

**Result:** DIAMOND's denoising manifold strongly resists external spatial control. Activation patching, pixel-space persistent guidance, and conditioning fine-tuning all fail to shift ball position. The world model's spatial layout is determined by frozen encoder/bottleneck weights that cannot be overridden from outside. This is a clean negative result about diffusion world model steerability.

### 3. Temporal sensitivity
Measures how DIAMOND's generation degrades when recent context frames are dropped.

**Result:** Dropping 1 of 4 context frames causes 21× larger frame deviation (0.002 → 0.047). Non-linear cliff effect — the model has learned temporal shortcuts that collapse without full history.

## Key findings

| Experiment | Finding |
|---|---|
| Probe R² by layer | Spatial encoding peaks at bottleneck/early decoder, near-zero at input/output layers |
| Pong vs Breakout probes | R²=0.548 vs R²=0.05 — encoding proportional to causal relevance |
| Steering resistance | All inference-time steering methods saturate at ~0.001–0.006 mean\|Δframe\| |
| Temporal sensitivity | 1-frame drop → 21× effect; 2-frame drop → 80× effect |
| Action masking | Masking all 4 action steps → 0.003 Δframe (modest, consistent) |
| Sigma perturbation | Compressing sigma schedule → 0.005 Δframe (largest at σ×0.25) |

## Setup

```bash
git clone https://github.com/eloialonso/diamond ~/diamond
git clone https://github.com/akumoli-debug/dream-perturbation
cd dream-perturbation
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install gymnasium[atari] ale-py autorom matplotlib huggingface_hub
AutoROM --accept-license
huggingface-cli download eloialonso/diamond --local-dir ~/diamond/pretrained
export DIAMOND_ROOT=~/diamond
export DIAMOND_CKPT=~/diamond/pretrained/atari_100k/models/Pong.pt
export PYTHONPATH="$DIAMOND_ROOT/src:./src:$PYTHONPATH"
```

## Running the experiments

```bash
# Step 1: Collect RAM-labelled frames
python3 src/step2_collect_ram_labels.py

# Step 2: Train linear probes on all U-Net layers
python3 src/step4_train_probes.py --game Pong
python3 src/step4_train_probes.py --game Breakout

# Step 3: Orthogonalize probe vector against confounds
python3 src/step4b_orthogonalize.py

# Step 4: Geometric steering experiment
python3 src/step5_geometric_steering_env.py

# Step 5: Fine-tune with ball_y conditioning
python3 src/finetune_conditioned_diamond.py

# Step 6: Temporal sensitivity + action masking + sigma perturbation
# (run inline — see results/option_b_results.json)
```

## Results

