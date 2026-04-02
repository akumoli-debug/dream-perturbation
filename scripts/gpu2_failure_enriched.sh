#!/bin/bash
# ============================================================
# GPU INSTANCE 2 — Failure-Enriched Training
# 
# Run this on your SECOND Vast.ai RTX 5090 instance.
# This trains DIAMOND with failure-enriched data.
#
# IMPORTANT: You must first collect and curate data on GPU 1,
# then transfer the curated dataset to this instance:
#
#   scp -r gpu1:~/diamond/experiment_data/ ~/diamond/experiment_data/
#
# Usage: bash gpu2_failure_enriched.sh [GAME]
# ============================================================

set -e

GAME=${1:-"BreakoutNoFrameskip-v4"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "  GPU 2 — FAILURE-ENRICHED EXPERIMENT"
echo "  Game: ${GAME}"
echo "  Started: $(date)"
echo "============================================================"

# ---- SETUP ----
echo ""
echo "[1/3] Setting up environment..."

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

conda create -n diamond python=3.10 -y 2>/dev/null || true
conda activate diamond

cd $HOME
[ ! -d "diamond" ] && git clone https://github.com/eloialonso/diamond.git
cd diamond
pip install -r requirements.txt
pip install lpips scikit-learn matplotlib seaborn 2>/dev/null

python -c "
import torch
assert torch.cuda.is_available(), 'No GPU!'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# ---- CHECK DATA ----
echo ""
echo "[1.5/3] Checking for curated dataset..."

if [ ! -d "experiment_data/failure_enriched/dataset/train" ]; then
    echo ""
    echo "ERROR: Failure-enriched dataset not found!"
    echo ""
    echo "You need to either:"
    echo "  a) Run collect_and_curate.py on this instance first:"
    echo "     python ~/learning-from-failure/scripts/collect_and_curate.py --game ${GAME}"
    echo ""
    echo "  b) Or transfer data from GPU 1:"
    echo "     scp -r gpu1:~/diamond/experiment_data/ ~/diamond/experiment_data/"
    echo ""
    
    echo "Running data collection locally..."
    cd $HOME
    [ ! -d "learning-from-failure" ] && git clone https://github.com/YOUR_USERNAME/learning-from-failure.git
    cd $HOME/diamond
    
    python $HOME/learning-from-failure/scripts/collect_and_curate.py \
        --game ${GAME} \
        --steps 100000
fi

# ---- TRAINING ----
echo ""
echo "[2/3] Training DIAMOND on FAILURE-ENRICHED data..."
echo "  This will take ~35 hours on RTX 5090 (~70 hours on RTX 4090)"
echo "  Started: $(date)"

cd $HOME/diamond

python src/main.py \
    static_dataset.path=experiment_data/failure_enriched/dataset \
    env.train.id=${GAME} \
    common.devices=0 \
    wandb.mode=disabled \
    training.compile_wm=True

echo "  Training completed: $(date)"

# ---- SAVE RESULTS ----
echo ""
echo "[3/3] Saving results..."

LATEST_OUTPUT=$(ls -td outputs/*/ | head -1)
RESULTS_DIR="$HOME/results/failure_enriched_${GAME}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
cp -r "$LATEST_OUTPUT/checkpoints" "$RESULTS_DIR/"
cp -r "experiment_data/curation_results.json" "$RESULTS_DIR/" 2>/dev/null || true
cp -r "experiment_data/failure_enriched/dataset/train/info.pt" "$RESULTS_DIR/dataset_info.pt" 2>/dev/null || true

echo ""
echo "============================================================"
echo "  FAILURE-ENRICHED TRAINING COMPLETE"
echo "  Results saved to: ${RESULTS_DIR}"
echo "  Finished: $(date)"
echo "============================================================"
