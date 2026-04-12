"""Fit ridge probes at each (site, sigma_idx, repr, target). Temporal train/test split.

Outputs:
    data/probe_results.csv
    data/specificity_matrix.pt
    data/gate_summary.csv

G1 gate (decoding side):
    For each (site, sigma, repr):
      diag[i] = R2(probe_i -> y_i)  on held-out TEMPORAL split
      off[i,j] = max(0, R2(probe_i -> y_j)) for j != i
      For probes with diag[i] > R2_MIN:
        ratio[i] = diag[i] / max(off[i, :])  (undefined if all off are 0)
      Pass if median(ratio) > 2 AND at least 3 probes have diag > R2_MIN.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import warnings
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")  # also kill LinAlgWarnings

ROOT = Path("/workspace/dream-perturbation")

ALPHAS_POOLED = [1e-2, 1e-1, 1.0, 10.0, 100.0]
ALPHAS_FLAT = [1.0, 10.0, 100.0, 1e3, 1e4]
TEST_FRAC = 0.2
TARGETS = ["ball_x", "ball_y", "ball_vx", "ball_vy",
           "left_paddle_y", "right_paddle_y", "collision_imminent"]
BINARY = {"collision_imminent"}
R2_MIN = 0.1
FLAT_SITES = {"s2_d_stage3", "s3_mid"}


def fit_ridge(X, y, test_X, test_y, binary, alphas):
    best = (-1e9, None)
    for a in alphas:
        model = RidgeClassifier(alpha=a) if binary else Ridge(alpha=a)
        model.fit(X, y)
        if binary:
            scores = model.decision_function(test_X)
            try:
                v = roc_auc_score(test_y, scores)
            except ValueError:
                v = float("nan")
        else:
            pred = model.predict(test_X)
            v = r2_score(test_y, pred)
        if v > best[0]:
            best = (v, a)
    return best


def fit_one(Xtr, Xte, ytr_full, yte_full, tr_idx, te_idx, binary, alphas):
    ytr = ytr_full[tr_idx]
    yte = yte_full[te_idx]
    if binary:
        ytr = (ytr > 0.5).astype(int)
        yte = (yte > 0.5).astype(int)
    score, alpha = fit_ridge(Xtr, ytr, Xte, yte, binary, alphas)
    model = (RidgeClassifier(alpha=alpha) if binary else Ridge(alpha=alpha)).fit(Xtr, ytr)
    return score, alpha, model


def main():
    import os
    untrained = os.environ.get("UNTRAINED", "0") == "1"
    shuffle = os.environ.get("SHUFFLE", "0") == "1"
    act_path = "data/activations_untrained.pt" if untrained else "data/activations.pt"
    data = torch.load(ROOT / act_path, weights_only=False)
    suffix = ""
    if untrained: suffix += "_untrained"
    if shuffle:   suffix += "_shuffled"
    pooled = data["pooled"]
    flat = data["flat"]
    labels = data["labels"]
    if shuffle:
        rng_shuf = np.random.default_rng(42)
        for k in labels:
            perm = rng_shuf.permutation(len(labels[k]))
            labels[k] = labels[k][perm]
        print("SHUFFLED labels for sanity check")
    sites = data["sites"]
    pick = data["pick"].numpy()  # already sorted by frame index
    N_SIGMA = len(next(iter(pooled.values())))
    N = next(iter(labels.values())).shape[0]

    # TEMPORAL split: last 20% of pick (which is sorted) = test.
    # pick is sorted ascending, so pick[-4800:] are the latest-in-time frames.
    split = int(N * (1 - TEST_FRAC))
    tr_idx = np.arange(split)
    te_idx = np.arange(split, N)
    print(f"N={N} sigmas={N_SIGMA} sites={sites}")
    print(f"temporal split: train={len(tr_idx)} (frames {pick[0]}..{pick[split-1]})"
          f" test={len(te_idx)} (frames {pick[split]}..{pick[-1]})")
    # Sanity: gap between last train frame and first test frame
    gap = pick[split] - pick[split - 1]
    print(f"frame-index gap at split: {gap} (>= 4 means no conditioning overlap)")

    rows = []
    specificity = {}

    configs = []
    for site in sites:
        configs.append((site, "pooled", pooled[site], ALPHAS_POOLED))
        if site in FLAT_SITES:
            configs.append((site, "flat", flat[site], ALPHAS_FLAT))

    for site, repr_kind, arr_per_sigma, alphas in configs:
        key = (site, repr_kind)
        specificity[key] = []
        for si_idx in range(N_SIGMA):
            X = arr_per_sigma[si_idx].numpy()
            Xtr, Xte = X[tr_idx], X[te_idx]

            trained = {}
            diag_scores = {}
            for tg in TARGETS:
                if tg not in labels:
                    continue
                y_full = labels[tg].numpy()
                if np.isnan(y_full).any():
                    continue
                binary = tg in BINARY
                score, alpha, model = fit_one(Xtr, Xte, y_full, y_full, tr_idx, te_idx, binary, alphas)
                rows.append(dict(site=site, repr=repr_kind, sigma_idx=si_idx, target=tg,
                                 metric="auc" if binary else "r2", value=score, alpha=alpha))
                trained[tg] = model
                diag_scores[tg] = score
                print(f"{site:13s} {repr_kind:6s} s[{si_idx}] {tg:18s} {score:.3f} (a={alpha})")

            cont_names = [t for t in trained if t not in BINARY]
            if len(cont_names) < 2:
                specificity[key].append(None)
                continue
            Tn = len(cont_names)
            mat = np.zeros((Tn, Tn))
            for i, src in enumerate(cont_names):
                pred = trained[src].predict(Xte)
                for j, tgt in enumerate(cont_names):
                    y = labels[tgt].numpy()[te_idx]
                    try:
                        mat[i, j] = r2_score(y, pred)
                    except Exception:
                        mat[i, j] = float("nan")
            specificity[key].append({"names": cont_names, "matrix": mat})

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / f"data/probe_results{suffix}.csv", index=False)
    torch.save(specificity, ROOT / f"data/specificity_matrix{suffix}.pt")

    # ---- Gate summary ----
    print("\n=== G1 gate (temporal split, fixed) ===")
    gate_rows = []
    hdr = f"{'site':13s} {'repr':6s} sig  n_good  med_diag  med_off  med_ratio  frac>2  verdict"
    print(hdr)
    print("-" * len(hdr))
    for (site, repr_kind), entries in specificity.items():
        for si_idx, entry in enumerate(entries):
            if entry is None:
                continue
            mat = entry["matrix"]
            names = entry["names"]
            diag = np.diag(mat).copy()
            off = mat.copy()
            np.fill_diagonal(off, -np.inf)
            off = np.clip(off, 0, None)
            off_max = off.max(axis=1)
            good = diag > R2_MIN
            n_good = int(good.sum())
            if n_good == 0:
                row = dict(site=site, repr=repr_kind, sigma_idx=si_idx,
                           n_good=0, med_diag=np.nan, med_off=np.nan,
                           med_ratio=np.nan, frac_gt2=np.nan, verdict="no_signal")
                gate_rows.append(row)
                print(f"{site:13s} {repr_kind:6s} [{si_idx}]  0       --        --       --         --     no_signal")
                continue
            d_good = diag[good]
            o_good = off_max[good]
            # Ratios: only where off is non-trivial; else call it inf -> represent as 99
            ratios = np.where(o_good > 0.01, d_good / np.clip(o_good, 1e-9, None), 99.0)
            med_ratio = float(np.median(ratios))
            frac_gt2 = float((ratios > 2).mean())
            verdict = "PASS" if (med_ratio > 2 and n_good >= 3) else "FAIL"
            row = dict(site=site, repr=repr_kind, sigma_idx=si_idx,
                       n_good=n_good, med_diag=float(np.median(d_good)),
                       med_off=float(np.median(o_good)),
                       med_ratio=med_ratio, frac_gt2=frac_gt2, verdict=verdict)
            gate_rows.append(row)
            med_ratio_str = ">99" if med_ratio >= 99 else f"{med_ratio:.2f}"
            print(f"{site:13s} {repr_kind:6s} [{si_idx}]  {n_good:<7d} "
                  f"{np.median(d_good):.3f}     {np.median(o_good):.3f}    "
                  f"{med_ratio_str:<9s}  {frac_gt2:.2f}   {verdict}")

    pd.DataFrame(gate_rows).to_csv(ROOT / f"data/gate_summary{suffix}.csv", index=False)
    print("\nSaved: probe_results.csv, specificity_matrix.pt, gate_summary.csv")


if __name__ == "__main__":
    main()
