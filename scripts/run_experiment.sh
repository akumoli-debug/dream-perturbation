#!/bin/bash
# ============================================================
# Run a DIAMOND training experiment
# Usage: 
#   bash run_experiment.sh baseline BreakoutNoFrameskip-v4
#   bash run_experiment.sh failure_enriched BreakoutNoFrameskip-v4
# ============================================================

set -e

VARIANT=${1:-"baseline"}  # "baseline" or "failure_enriched"
GAME=${2:-"BreakoutNoFrameskip-v4"}

echo "============================================"
echo "  Training DIAMOND — ${VARIANT} variant"
echo "  Game: ${GAME}"
echo "============================================"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate diamond

cd ~/diamond

# Check if data has been collected and curated
if [ ! -d "experiment_data/${VARIANT}/dataset/train" ]; then
    echo ""
    echo "ERROR: Dataset not found at experiment_data/${VARIANT}/dataset/train"
    echo "Run collect_and_curate.py first!"
    echo ""
    exit 1
fi

echo ""
echo "Dataset: experiment_data/${VARIANT}/dataset/"
echo "Starting training..."
echo ""

# Train using DIAMOND's static dataset mode
# This tells DIAMOND to use our pre-collected dataset instead of collecting its own
python src/main.py \
    static_dataset.path=experiment_data/${VARIANT}/dataset \
    env.train.id=${GAME} \
    common.devices=0 \
    wandb.mode=disabled \
    training.compile_wm=True

echo ""
echo "============================================"
echo "  Training complete: ${VARIANT}"
echo "  Results in: outputs/ (latest directory)"
echo "============================================"
