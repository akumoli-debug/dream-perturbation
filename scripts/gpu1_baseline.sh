#!/bin/bash
# ============================================================
# GPU INSTANCE 1 — Baseline Training
# 
# Run this on your FIRST Vast.ai RTX 5090 instance.
# This trains DIAMOND with standard (unmodified) data.
#
# Steps:
#   1. Setup environment
#   2. Collect 100k steps of gameplay
#   3. Create baseline + failure-enriched datasets  
#   4. Train DIAMOND on BASELINE data
#   5. Run final evaluation (100 episodes)
#
# Usage: bash gpu1_baseline.sh [GAME]
# Default game: BreakoutNoFrameskip-v4
# ============================================================

set -e

GAME=${1:-"BreakoutNoFrameskip-v4"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "  GPU 1 — BASELINE EXPERIMENT"
echo "  Game: ${GAME}"
echo "  Started: $(date)"
echo "============================================================"

# ---- SETUP ----
echo ""
echo "[1/4] Setting up environment..."

# Install miniconda if needed
if ! command -v conda &> /dev/null; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    export PATH="$HOME/miniconda3/bin:$PATH"
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    source ~/.bashrc
else
    eval "$(conda shell.bash hook)"
fi

# Create env
conda create -n diamond python=3.10 -y 2>/dev/null || true
conda activate diamond

# Clone DIAMOND
cd $HOME
[ ! -d "diamond" ] && git clone https://github.com/eloialonso/diamond.git
cd diamond
pip install -r requirements.txt
pip install lpips scikit-learn matplotlib seaborn 2>/dev/null

# Clone our project
cd $HOME
[ ! -d "learning-from-failure" ] && git clone https://github.com/YOUR_USERNAME/learning-from-failure.git
# If not on GitHub yet, you'll scp the files instead

# Verify GPU
python -c "
import torch
assert torch.cuda.is_available(), 'No GPU!'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# ---- DATA COLLECTION ----
echo ""
echo "[2/4] Collecting and curating data..."
cd $HOME/diamond

python $HOME/learning-from-failure/scripts/collect_and_curate.py \
    --game ${GAME} \
    --steps 100000

# ---- TRAINING ----
echo ""
echo "[3/4] Training DIAMOND on BASELINE data..."
echo "  This will take ~35 hours on RTX 5090 (~70 hours on RTX 4090)"
echo "  Started: $(date)"

python src/main.py \
    static_dataset.path=experiment_data/baseline/dataset \
    env.train.id=${GAME} \
    common.devices=0 \
    wandb.mode=disabled \
    training.compile_wm=True

echo "  Training completed: $(date)"

# ---- SAVE RESULTS ----
echo ""
echo "[4/4] Saving results..."

# Find the latest output directory
LATEST_OUTPUT=$(ls -td outputs/*/ | head -1)

# Copy to a clearly named directory
RESULTS_DIR="$HOME/results/baseline_${GAME}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
cp -r "$LATEST_OUTPUT/checkpoints" "$RESULTS_DIR/"
cp -r "experiment_data/curation_results.json" "$RESULTS_DIR/"
cp -r "experiment_data/baseline/dataset/train/info.pt" "$RESULTS_DIR/dataset_info.pt" 2>/dev/null || true

echo ""
echo "============================================================"
echo "  BASELINE TRAINING COMPLETE"
echo "  Results saved to: ${RESULTS_DIR}"
echo "  Finished: $(date)"
echo "============================================================"
