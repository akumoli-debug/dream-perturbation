#!/usr/bin/env bash
# =============================================================================
# run_dream_perturbation.sh
# =============================================================================
# One-shot script to run the full Dream Perturbation experiment on a fresh
# Vast.ai instance (or any Linux machine) with a PyTorch 2.11 CUDA template.
#
# What this script does:
#   1. Detects environment (conda vs venv vs bare Python)
#   2. Clones the DIAMOND repository
#   3. Installs all dependencies (DIAMOND + matplotlib + huggingface_hub)
#   4. Applies PyTorch 2.11 compatibility patches to DIAMOND source
#   5. Downloads the pretrained Breakout model from HuggingFace
#   6. Runs the full dream_perturbation_experiment.py
#   7. Saves results/ and figures/
#
# Usage (from any directory):
#   bash /path/to/run_dream_perturbation.sh [OPTIONS]
#
# Environment variables you can override:
#   DIAMOND_ROOT        — path to clone/find DIAMOND (default: ./diamond)
#   EXPERIMENT_ROOT     — path to this project's root  (default: auto-detected)
#   RESULTS_DIR         — where to write results (default: EXPERIMENT_ROOT/results)
#   NUM_EPOCHS          — training epochs per agent (default: 50)
#   TRAIN_STEPS         — actor-critic steps per epoch (default: 200)
#   EVAL_EPISODES       — real-env evaluation episodes (default: 10)
#   WM_HORIZON          — dream horizon steps (default: 50)
#   BATCH_SIZE          — parallel envs in world model (default: 16)
#   INIT_COLLECT_STEPS  — initial real-env collection steps (default: 1000)
#   SEED                — random seed (default: 42)
#   FAST_MODE           — set to "1" for quick smoke-test (default: 0)
#   DEVICE              — PyTorch device (default: auto)
#   CONDA_ENV_NAME      — conda env to create/use (default: diamond)
#   PYTHON_VERSION      — Python version for new conda env (default: 3.10)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colour output helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[OK]${RESET}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }
banner()  { echo -e "\n${BOLD}${CYAN}=== $* ===${RESET}\n"; }

# ---------------------------------------------------------------------------
# Resolve script location (works with bash run_..., source, and symlinks)
# ---------------------------------------------------------------------------
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(dirname "$SCRIPT_DIR")}"  # parent of scripts/

info "Script path:       $SCRIPT_PATH"
info "Experiment root:   $EXPERIMENT_ROOT"

# ---------------------------------------------------------------------------
# Default configuration (overridable via env vars)
# ---------------------------------------------------------------------------
DIAMOND_ROOT="${DIAMOND_ROOT:-${EXPERIMENT_ROOT}/diamond}"
RESULTS_DIR="${RESULTS_DIR:-${EXPERIMENT_ROOT}/results}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
TRAIN_STEPS="${TRAIN_STEPS:-200}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
WM_HORIZON="${WM_HORIZON:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
INIT_COLLECT_STEPS="${INIT_COLLECT_STEPS:-1000}"
SEED="${SEED:-42}"
FAST_MODE="${FAST_MODE:-0}"
DEVICE="${DEVICE:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-diamond}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

EXPERIMENT_SCRIPT="${EXPERIMENT_ROOT}/scripts/dream_perturbation_experiment.py"
LFF_SRC="${EXPERIMENT_ROOT}/src"

# ---------------------------------------------------------------------------
# Helper: check if a command exists
# ---------------------------------------------------------------------------
has_cmd() { command -v "$1" &>/dev/null; }

# ---------------------------------------------------------------------------
# Step 1: Environment setup
# ---------------------------------------------------------------------------
banner "Step 1: Environment Setup"

setup_python_env() {
    if has_cmd conda; then
        info "conda detected — using conda environment '${CONDA_ENV_NAME}'"

        # Check if env already exists
        if conda env list | grep -qE "^${CONDA_ENV_NAME}\s"; then
            info "Conda env '${CONDA_ENV_NAME}' already exists"
        else
            info "Creating conda env '${CONDA_ENV_NAME}' with Python ${PYTHON_VERSION}…"
            conda create -y -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}"
            success "Conda env created"
        fi

        # Source conda so we can activate
        CONDA_BASE="$(conda info --base)"
        # shellcheck disable=SC1091
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV_NAME}"
        success "Conda env '${CONDA_ENV_NAME}' activated"
        PYTHON_CMD="python"

    elif has_cmd python3; then
        # Bare Python / Vast.ai PyTorch template — use a venv
        VENV_DIR="${EXPERIMENT_ROOT}/.venv"
        if [ ! -d "$VENV_DIR" ]; then
            info "Creating venv at ${VENV_DIR}…"
            python3 -m venv "$VENV_DIR"
        fi
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
        success "Venv activated"
        PYTHON_CMD="python"

    else
        error "Neither conda nor python3 found. Install Python 3.10+ first."
    fi

    info "Python: $(${PYTHON_CMD} --version)"
}

setup_python_env

# ---------------------------------------------------------------------------
# Step 2: Clone DIAMOND
# ---------------------------------------------------------------------------
banner "Step 2: Clone DIAMOND"

if [ -d "${DIAMOND_ROOT}/.git" ]; then
    info "DIAMOND already cloned at ${DIAMOND_ROOT}"
else
    info "Cloning DIAMOND from GitHub…"
    git clone https://github.com/eloialonso/diamond.git "${DIAMOND_ROOT}"
    success "DIAMOND cloned to ${DIAMOND_ROOT}"
fi

# ---------------------------------------------------------------------------
# Step 3: Install dependencies
# ---------------------------------------------------------------------------
banner "Step 3: Install Dependencies"

install_deps() {
    # Check if PyTorch is already installed (Vast.ai template often has it)
    if ${PYTHON_CMD} -c "import torch; print(torch.__version__)" 2>/dev/null; then
        TORCH_VER="$(${PYTHON_CMD} -c 'import torch; print(torch.__version__)')"
        info "PyTorch ${TORCH_VER} already installed"
    else
        warn "PyTorch not found — installing CPU version (for GPU, pre-install PyTorch manually)"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi

    # DIAMOND requirements
    info "Installing DIAMOND requirements…"
    pip install -r "${DIAMOND_ROOT}/requirements.txt"

    # Extra packages needed by this experiment
    info "Installing extra packages (matplotlib, huggingface_hub, omegaconf, hydra)…"
    pip install \
        matplotlib \
        "huggingface_hub>=0.20" \
        "omegaconf>=2.3" \
        "hydra-core>=1.3" \
        numpy \
        tqdm

    success "All dependencies installed"
}

install_deps

# ---------------------------------------------------------------------------
# Step 4: Apply PyTorch 2.11 patches to DIAMOND source
# ---------------------------------------------------------------------------
banner "Step 4: Apply Compatibility Patches"

DIAMOND_SRC="${DIAMOND_ROOT}/src"

apply_patches() {
    # -------------------------------------------------------------------------
    # Patch A: BatchSampler.__init__ — super().__init__(dataset) ->
    #          super().__init__()
    #
    # PyTorch >= 2.11 changed torch.utils.data.Sampler.__init__ to accept
    # no positional arguments (the `dataset` arg was removed).
    # -------------------------------------------------------------------------
    BATCH_SAMPLER_FILE="${DIAMOND_SRC}/data/batch_sampler.py"
    if [ -f "$BATCH_SAMPLER_FILE" ]; then
        if grep -q "super().__init__(dataset)" "$BATCH_SAMPLER_FILE"; then
            info "Patching BatchSampler.__init__ for PyTorch 2.11+…"
            sed -i 's/super().__init__(dataset)/super().__init__()/g' "$BATCH_SAMPLER_FILE"
            success "BatchSampler patch applied"
        else
            info "BatchSampler already patched (or different version)"
        fi
    else
        warn "BatchSampler file not found at ${BATCH_SAMPLER_FILE}, skipping"
    fi

    # -------------------------------------------------------------------------
    # Patch B: torch.load weights_only — add weights_only=False to all
    #          torch.load calls in DIAMOND's source.
    #
    # PyTorch 2.6+ raises a FutureWarning (and in 2.11+ errors) when
    # loading checkpoints that contain non-tensor objects without
    # explicitly setting weights_only=False.
    # -------------------------------------------------------------------------
    info "Patching torch.load calls to use weights_only=False…"
    PATCHED=0
    while IFS= read -r -d '' py_file; do
        if grep -q "torch\.load(" "$py_file"; then
            # Only patch calls that don't already have weights_only
            if ! grep -q "weights_only" "$py_file"; then
                # Add weights_only=False before the closing paren on torch.load lines
                sed -i \
                    's/torch\.load(\(.*\))/torch.load(\1, weights_only=False)/g' \
                    "$py_file"
                PATCHED=$((PATCHED + 1))
                info "  Patched: $py_file"
            fi
        fi
    done < <(find "${DIAMOND_SRC}" -name "*.py" -print0)
    success "torch.load patched in ${PATCHED} files"

    # -------------------------------------------------------------------------
    # Patch C: OmegaConf resolver in main.py / play.py
    #
    # Ensure OmegaConf.register_new_resolver("eval", eval, replace=True)
    # is present.  The experiment script handles this at runtime as well,
    # but patching the source avoids issues if DIAMOND's scripts are called.
    # -------------------------------------------------------------------------
    for f in "${DIAMOND_SRC}/main.py" "${DIAMOND_SRC}/play.py"; do
        if [ -f "$f" ]; then
            if grep -q 'register_new_resolver("eval"' "$f"; then
                # Add replace=True if it's missing
                if ! grep -q 'replace=True' "$f"; then
                    sed -i \
                        's/register_new_resolver("eval", eval)/register_new_resolver("eval", eval, replace=True)/g' \
                        "$f"
                    info "  Added replace=True to OmegaConf resolver in $(basename $f)"
                fi
            fi
        fi
    done
    success "OmegaConf resolver patch checked"
}

apply_patches

# ---------------------------------------------------------------------------
# Step 5: Download pretrained Breakout checkpoint
# ---------------------------------------------------------------------------
banner "Step 5: Download Pretrained Breakout Checkpoint"

download_checkpoint() {
    CKPT_CACHE_DIR="${HOME}/.cache/huggingface/hub"
    CKPT_SUBPATH="models--eloialonso--diamond/snapshots"

    # Check if already cached by huggingface_hub
    if find "${CKPT_CACHE_DIR}" -name "Breakout.pt" 2>/dev/null | grep -q "Breakout.pt"; then
        CKPT_PATH="$(find "${CKPT_CACHE_DIR}" -name "Breakout.pt" | head -1)"
        info "Checkpoint already cached at: ${CKPT_PATH}"
    else
        info "Downloading Breakout.pt from HuggingFace (eloialonso/diamond)…"
        ${PYTHON_CMD} - <<'PYEOF'
from huggingface_hub import hf_hub_download
import os
path = hf_hub_download(
    repo_id="eloialonso/diamond",
    filename="atari_100k/models/Breakout.pt",
)
print(f"Downloaded to: {path}")
PYEOF
        success "Breakout.pt downloaded"
    fi
}

download_checkpoint

# ---------------------------------------------------------------------------
# Step 6: Create results directory
# ---------------------------------------------------------------------------
banner "Step 6: Prepare Results Directory"
mkdir -p "${RESULTS_DIR}/figures"
info "Results will be written to: ${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# Step 7: Run the experiment
# ---------------------------------------------------------------------------
banner "Step 7: Run Dream Perturbation Experiment"

# Build argument list
ARGS=(
    --diamond-root "${DIAMOND_ROOT}"
    --output-dir   "${RESULTS_DIR}"
    --num-epochs   "${NUM_EPOCHS}"
    --train-steps-per-epoch "${TRAIN_STEPS}"
    --eval-episodes "${EVAL_EPISODES}"
    --wm-horizon   "${WM_HORIZON}"
    --batch-size   "${BATCH_SIZE}"
    --init-collect-steps "${INIT_COLLECT_STEPS}"
    --seed         "${SEED}"
)

if [ -n "${DEVICE}" ]; then
    ARGS+=(--device "${DEVICE}")
fi

if [ "${FAST_MODE}" = "1" ]; then
    ARGS+=(--fast)
    warn "FAST_MODE=1 — running reduced epochs for smoke-test only"
fi

info "Experiment arguments: ${ARGS[*]}"
info "Starting experiment at $(date)…"

START_TIME="$SECONDS"

"${PYTHON_CMD}" "${EXPERIMENT_SCRIPT}" "${ARGS[@]}"

END_TIME="$SECONDS"
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS_REM=$((ELAPSED % 60))

# ---------------------------------------------------------------------------
# Step 8: Post-run summary
# ---------------------------------------------------------------------------
banner "Step 8: Results Summary"

RESULTS_JSON="${RESULTS_DIR}/dream_perturbation_results.json"
COMPARISON_FIG="${RESULTS_DIR}/figures/dream_perturbation_comparison.png"
HEATMAP_FIG="${RESULTS_DIR}/figures/dream_robustness.png"
CURVES_FIG="${RESULTS_DIR}/figures/training_curves.png"

success "Experiment completed in ${MINUTES}m ${SECONDS_REM}s"
echo ""
echo -e "${BOLD}Output files:${RESET}"

for f in "$RESULTS_JSON" "$COMPARISON_FIG" "$HEATMAP_FIG" "$CURVES_FIG"; do
    if [ -f "$f" ]; then
        SIZE="$(du -sh "$f" | cut -f1)"
        echo -e "  ${GREEN}✓${RESET} $f  (${SIZE})"
    else
        echo -e "  ${YELLOW}?${RESET} $f  (not found — check experiment logs)"
    fi
done

echo ""
info "To view the JSON results:"
echo "    cat ${RESULTS_JSON} | python -m json.tool | head -60"

echo ""
info "To copy figures to your local machine:"
echo "    scp -r user@VAST_IP:${RESULTS_DIR}/figures/ ./dream_perturbation_figures/"

# ---------------------------------------------------------------------------
# Print key metrics from JSON if available
# ---------------------------------------------------------------------------
if [ -f "$RESULTS_JSON" ] && has_cmd python3; then
    echo ""
    echo -e "${BOLD}Key metrics (real Atari return):${RESET}"
    ${PYTHON_CMD} - <<PYEOF
import json, sys
with open("${RESULTS_JSON}") as f:
    r = json.load(f)
real = r.get("real_env", {})
for agent, stats in real.items():
    mean = stats.get("mean_return", 0)
    std  = stats.get("std_return", 0)
    eps  = stats.get("num_episodes", 0)
    print(f"  {agent:<22} mean={mean:7.2f}  std={std:6.2f}  ({eps} episodes)")
PYEOF
fi

echo ""
success "Done! Results saved to ${RESULTS_DIR}"
