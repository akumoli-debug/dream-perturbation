"""Refit the C1 ridge probe at s3_mid/pooled/sigma[2] for ball_y.

Filters ball_y=0 sentinel (ball off-screen between points, ~37% of
random-policy frames). With filter, R2 matches probe_results.csv (0.71).
"""
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ROOT = Path("/workspace/dream-perturbation")
SITE, SIGMA_IDX, TARGET, ALPHA = "s3_mid", 2, "ball_y", 0.10
SENTINEL_FLOOR = 1.0
TEST_FRAC = 0.2
OUT = ROOT / "data/steering_vectors/c2_pilot_s3mid_sigma2_bally.pt"


def main():
    d = torch.load(ROOT / "data/activations.pt", weights_only=False)
    X = d["pooled"][SITE][SIGMA_IDX].numpy().astype(np.float32)
    y = d["labels"][TARGET].numpy().astype(np.float32)

    mask = (y > SENTINEL_FLOOR) & (~np.isnan(y))
    X, y = X[mask], y[mask]
    print(f"after sentinel filter: N={len(y)}  (dropped {(~mask).sum()})")

    N = len(y); n_test = int(N * TEST_FRAC)
    Xtr, Xte = X[:-n_test], X[-n_test:]
    ytr, yte = y[:-n_test], y[-n_test:]

    model = Ridge(alpha=ALPHA).fit(Xtr, ytr)
    r2 = r2_score(yte, model.predict(Xte))
    print(f"R2 test: {r2:.4f}  (CSV: 0.7147)")

    coef = model.coef_.astype(np.float32)
    sigma_norm = float(np.linalg.norm(coef))
    direction = coef / sigma_norm

    out = {
        "direction": torch.from_numpy(direction),
        "coef_raw": torch.from_numpy(coef),
        "intercept": float(model.intercept_),
        "sigma_norm": sigma_norm,
        "r2_test": float(r2),
        "meta": {
            "site": SITE, "sigma_idx": SIGMA_IDX, "target": TARGET,
            "alpha": ALPHA, "sentinel_floor": SENTINEL_FLOOR,
            "N_train": len(ytr), "N_test": len(yte),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT)
    print(f"saved {OUT}")
    print(f"  direction shape: {tuple(out['direction'].shape)}")
    print(f"  sigma_norm: {sigma_norm:.2f}")
    top5 = np.argsort(np.abs(direction))[-5:][::-1]
    print(f"  top-5 channels: {top5.tolist()}")
    print(f"  top-5 weights:  {direction[top5].round(3).tolist()}")


if __name__ == "__main__":
    main()
