#!/bin/bash
# ============================================================
# GPU Instance Setup Script — Run this FIRST on each Vast.ai instance
# Usage: bash setup_gpu.sh
# ============================================================

set -e

echo "============================================"
echo "  Learning From Failure — GPU Setup"
echo "============================================"

# 1. Verify GPU
echo ""
echo "[1/6] Checking GPU..."
nvidia-smi
echo ""

# 2. Install miniconda if not present
if ! command -v conda &> /dev/null; then
    echo "[2/6] Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    export PATH="$HOME/miniconda3/bin:$PATH"
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    source ~/.bashrc
else
    echo "[2/6] Conda already installed"
    eval "$(conda shell.bash hook)"
fi

# 3. Create environment
echo "[3/6] Creating conda environment..."
conda create -n diamond python=3.10 -y
conda activate diamond

# 4. Clone DIAMOND
echo "[4/6] Cloning DIAMOND..."
cd $HOME
if [ ! -d "diamond" ]; then
    git clone https://github.com/eloialonso/diamond.git
fi
cd diamond

# 5. Install dependencies
echo "[5/6] Installing dependencies..."
pip install -r requirements.txt
pip install lpips scikit-learn matplotlib seaborn

# 6. Quick test
echo "[6/6] Testing PyTorch + CUDA..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print('Setup complete!')
"

echo ""
echo "============================================"
echo "  Setup complete! Next steps:"
echo "  1. Copy src/ files from learning-from-failure repo"
echo "  2. Run: bash run_experiment.sh [baseline|failure_enriched]"
echo "============================================"
