"""Collect activations at 6 sites x 3 sigma steps over a pong frame dataset.

Inputs: data/pong_dataset/pong_dataset.npz
    frames:  (T, 210, 160, 3) uint8
    actions: (T,) int64
    labels:  (T, 7) float32
    label_keys: (7,) U18
Output: data/activations.pt
"""
import sys
import time
from pathlib import Path
import numpy as np
import cv2
import torch

ROOT = Path("/workspace/dream-perturbation")
sys.path.insert(0, str(ROOT))
from src.diamond_loader import load_denoiser
from src.diamond_hooks import ActivationCapture, SITES

N_SAMPLES = 24000
BATCH = 32
SIGMA_MIN, SIGMA_MAX, RHO, N_SIGMA = 2e-3, 5.0, 7, 3
FLAT_SITES = {"s2_d_stage3", "s3_mid"}
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)


def build_sigmas(n, smin, smax, rho, device):
    min_inv_rho = smin ** (1.0 / rho)
    max_inv_rho = smax ** (1.0 / rho)
    l = torch.linspace(0, 1, n, device=device)
    return (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho


def preprocess_frames(frames):
    out = np.empty((frames.shape[0], 64, 64, 3), dtype=np.uint8)
    for i in range(frames.shape[0]):
        out[i] = cv2.resize(frames[i], (64, 64), interpolation=cv2.INTER_AREA)
    out = out.astype(np.float32).transpose(0, 3, 1, 2) / 127.5 - 1.0
    return out


def main():
    device = "cuda"
    import os
    untrained = os.environ.get("UNTRAINED", "0") == "1"
    denoiser, cfg = load_denoiser(untrained=untrained)
    denoiser.eval()
    n = cfg.denoiser.inner_model.num_steps_conditioning

    ds = np.load(ROOT / "data/pong_dataset/pong_dataset.npz")
    frames_raw = ds["frames"]
    act_all = ds["actions"]
    labels_arr = ds["labels"]
    label_keys = [str(k) for k in ds["label_keys"]]
    print("label_keys:", label_keys)
    T = frames_raw.shape[0]

    label_idx = {k: i for i, k in enumerate(label_keys)}
    ball_x_col = label_idx["ball_x"]
    valid_at_t = ~np.isnan(labels_arr[:, ball_x_col])
    valid_at_tp1 = np.zeros(T, dtype=bool)
    valid_at_tp1[:-1] = valid_at_t[1:]

    eligible = np.where(valid_at_tp1)[0]
    eligible = eligible[(eligible >= n - 1) & (eligible < T - 1)]
    print("eligible:", len(eligible), "/", T)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(eligible, size=min(N_SAMPLES, len(eligible)), replace=False)
    pick.sort()
    N = len(pick)
    print("picked N =", N)

    needed_set = set()
    for p in pick:
        for j in range(p - n + 1, p + 2):
            needed_set.add(int(j))
    needed = np.array(sorted(needed_set))
    print("preprocessing", len(needed), "frames...")
    t0 = time.time()
    frames_64 = preprocess_frames(frames_raw[needed])
    idx_map = {int(j): k for k, j in enumerate(needed)}
    print("preprocess", round(time.time() - t0, 1), "s, ram", round(frames_64.nbytes / 1e9, 2), "GB")

    labels_tp1 = labels_arr[pick + 1]
    labels_out = {k: torch.from_numpy(labels_tp1[:, i].copy()) for k, i in label_idx.items()}

    dummy_noisy = torch.randn(1, 3, 64, 64, device=device)
    dummy_obs = torch.randn(1, n * 3, 64, 64, device=device)
    dummy_act = torch.zeros(1, n, dtype=torch.long, device=device)
    with ActivationCapture(denoiser) as cap, torch.no_grad():
        _ = denoiser.inner_model(dummy_noisy, torch.zeros(1, device=device), dummy_obs, dummy_act)
    site_shapes = {k: tuple(v.shape[1:]) for k, v in cap.acts.items()}
    print("site shapes:", site_shapes)

    pooled = {name: [torch.zeros(N, shape[0]) for _ in range(N_SIGMA)]
              for name, shape in site_shapes.items()}
    flat = {name: [torch.zeros(N, int(np.prod(shape))) for _ in range(N_SIGMA)]
            for name, shape in site_shapes.items() if name in FLAT_SITES}

    sigmas = build_sigmas(N_SIGMA, SIGMA_MIN, SIGMA_MAX, RHO, device)
    print("sigmas:", sigmas.cpu().tolist())

    t0 = time.time()
    for b0 in range(0, N, BATCH):
        b1 = min(b0 + BATCH, N)
        bs = b1 - b0
        idx_batch = pick[b0:b1]

        cond_list = []
        tgt_list = []
        act_list = []
        for ii in idx_batch:
            cond_frames = [frames_64[idx_map[int(ii - j)]] for j in reversed(range(n))]
            cond_list.append(np.concatenate(cond_frames, axis=0))
            tgt_list.append(frames_64[idx_map[int(ii + 1)]])
            act_list.append(act_all[ii - n + 1:ii + 1])
        cond = np.stack(cond_list, axis=0)
        target = np.stack(tgt_list, axis=0)
        acts = np.stack(act_list, axis=0)

        obs_t = torch.from_numpy(cond).to(device)
        target_t = torch.from_numpy(target).to(device)
        act_t = torch.from_numpy(acts).to(device).long()

        for si_idx in range(N_SIGMA):
            sigma = sigmas[si_idx].expand(bs)
            noisy = denoiser.apply_noise(target_t, sigma, denoiser.cfg.sigma_offset_noise)
            cs = denoiser.compute_conditioners(sigma)
            rescaled_obs = obs_t / denoiser.cfg.sigma_data
            rescaled_noise = noisy * cs.c_in
            with ActivationCapture(denoiser) as cap, torch.no_grad():
                _ = denoiser.inner_model(rescaled_noise, cs.c_noise.squeeze(), rescaled_obs, act_t)
            for name, tensor in cap.acts.items():
                pooled[name][si_idx][b0:b1] = tensor.mean(dim=(2, 3)).cpu()
                if name in FLAT_SITES:
                    flat[name][si_idx][b0:b1] = tensor.reshape(bs, -1).cpu()

        if (b0 // BATCH) % 50 == 0:
            dt = time.time() - t0
            rate = b1 / max(dt, 1e-9)
            print("[", b1, "/", N, "] elapsed", round(dt, 1), "s rate", round(rate, 1), "fr/s")

    out = {
        "pooled": pooled,
        "flat": flat,
        "labels": labels_out,
        "sigmas": sigmas.cpu(),
        "sites": [s[0] for s in SITES],
        "pick": torch.from_numpy(pick),
    }
    out_path = ROOT / ("data/activations_untrained.pt" if untrained else "data/activations.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print("saved", out_path)


if __name__ == "__main__":
    main()
