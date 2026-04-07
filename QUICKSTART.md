# Quick Start — Do This RIGHT NOW

## Step 1: Rent 2x RTX 5090 on Vast.ai

1. Go to https://vast.ai
2. Create account, add $50 credit
3. Click "Search" → filter by:
   - GPU: RTX 5090
   - Min VRAM: 32 GB
   - Min Disk: 50 GB
   - Sort by: price (low to high)
4. Rent TWO instances (cheapest reliable ones, ~$0.37/hr each)
5. Choose a Docker image: `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel`
   - Or just use the default Ubuntu image

## Step 2: Upload project files to BOTH instances

SSH into each instance and run:

```bash
# On each GPU instance:
cd ~
git clone https://github.com/eloialonso/diamond.git

# Upload the dream-perturbation project files
# Option A: If you push to GitHub first
git clone https://github.com/YOUR_USERNAME/dream-perturbation.git

# Option B: scp from your laptop
# From your laptop terminal:
# scp -P <PORT> -r dream-perturbation/ root@<VAST_IP>:~/
```

## Step 3: Run data collection on GPU 1

SSH into GPU 1:
```bash
# Install dependencies
cd ~/diamond
pip install -r requirements.txt
pip install lpips scikit-learn matplotlib seaborn

# Collect and curate data (~15-30 min)
cd ~/diamond
export DIAMOND_DIR=~/diamond
python ~/dream-perturbation/scripts/collect_and_curate.py --game BreakoutNoFrameskip-v4
```

This creates:
- `experiment_data/baseline/dataset/` — unmodified data
- `experiment_data/failure_enriched/dataset/` — failure episodes duplicated

## Step 4: Copy curated data to GPU 2

From GPU 1, scp the curated data to GPU 2:
```bash
# From GPU 1 (replace GPU2_IP and GPU2_PORT with GPU 2's SSH details):
scp -P <GPU2_PORT> -r ~/diamond/experiment_data/ root@<GPU2_IP>:~/diamond/experiment_data/
```

Or from your laptop, download from GPU 1 then upload to GPU 2.

## Step 5: Start training on BOTH GPUs simultaneously

**GPU 1** (baseline):
```bash
cd ~/diamond
python src/main.py \
    static_dataset.path=experiment_data/baseline/dataset \
    env.train.id=BreakoutNoFrameskip-v4 \
    common.devices=0 \
    wandb.mode=disabled \
    training.compile_wm=True
```

**GPU 2** (failure-enriched):
```bash
cd ~/diamond
python src/main.py \
    static_dataset.path=experiment_data/failure_enriched/dataset \
    env.train.id=BreakoutNoFrameskip-v4 \
    common.devices=0 \
    wandb.mode=disabled \
    training.compile_wm=True
```

## Step 6: Wait ~35 hours (5090) or ~70 hours (4090)

Training will run. You can monitor with:
```bash
# Check if it's still running
ps aux | grep python

# Check GPU utilization
nvidia-smi

# Check latest output
ls -la outputs/
```

The process is checkpointed — if it crashes, you can resume:
```bash
cd ~/diamond/outputs/YYYY-MM-DD/hh-mm-ss/
./scripts/resume.sh
```

## Step 7: Download results

After both finish:
```bash
# From your laptop, download the trained models
# GPU 1 (baseline):
scp -P <PORT> -r root@<GPU1_IP>:~/diamond/outputs/ ./results/baseline/

# GPU 2 (failure-enriched):
scp -P <PORT> -r root@<GPU2_IP>:~/diamond/outputs/ ./results/failure_enriched/
```

## Step 8: Run evaluation


---

## Estimated Timeline

| Time | What happens |
|------|-------------|
| Now (11:30 PM) | Rent GPUs, upload files |
| +30 min | Data collection + curation done |
| +1 hr | Both training runs started |
| +36 hrs (Thu ~noon) | Training finishes (5090) |
| Thu afternoon | Run evaluation, generate figures |
| Friday | Write up results, clean repo |
| Sat-Sun | Polish, review |
| Monday Apr 7 | Send to Pim |

## Cost Estimate

- 2 × RTX 5090 × 36 hrs × $0.37/hr = **~$27**
- Buffer for setup/debugging: ~$5
- **Total: ~$32**

## If Something Goes Wrong

- **Out of disk**: DIAMOND datasets are small (~2-3 GB). Should be fine with 50 GB.
- **CUDA OOM**: Unlikely — DIAMOND only needs 12 GB VRAM, 5090 has 32 GB.
- **Training crashes**: Use `./scripts/resume.sh` from the output directory.
- **Can't find 5090**: Use RTX 4090 instead. Same scripts, just takes ~70 hrs.
