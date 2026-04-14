"""Build data/pong_dataset/pong_dataset.npz for the C1 probe sweep.

Produces:
    frames:     (T, 210, 160, 3) uint8  — native Atari RGB
    actions:    (T,) int64
    labels:     (T, 7) float32          — per-frame labels (NaN allowed)
    label_keys: (7,) U18                — name order

Labels:
    ball_x, ball_y, ball_vx, ball_vy, left_paddle_y, right_paddle_y, collision_imminent

Pong RAM offsets (ALE, verified against stella-pong docs):
    49  -> ball_x   (0 when ball off-screen / between points)
    54  -> ball_y
    50  -> CPU paddle y    (left side)
    51  -> player paddle y (right side)

Run:
    python -m src.probes.build_pong_dataset
"""
import os
from pathlib import Path
import numpy as np
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

ROOT = Path("/workspace/dream-perturbation")
OUT_DIR = ROOT / "data" / "pong_dataset"
OUT_PATH = OUT_DIR / "pong_dataset.npz"

TARGET_FRAMES = 30000          # C1 uses 24k after filtering
MAX_STEPS_PER_EP = 3000
COLLISION_X_THRESH = 20        # pixels from right paddle x-column
COLLISION_VX_SIGN = 1          # ball approaching right paddle (vx > 0)
SEED = 0

LABEL_KEYS = np.array([
    "ball_x", "ball_y", "ball_vx", "ball_vy",
    "left_paddle_y", "right_paddle_y", "collision_imminent",
], dtype="U18")


def make_env():
    # No frameskip wrapper — we want every frame, native resolution, RGB.
    return gym.make(
        "ALE/Pong-v5",
        obs_type="rgb",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
    )


def read_ram_labels(ram, prev_ball):
    bx = int(ram[49])
    by = int(ram[54])
    lp = int(ram[50])
    rp = int(ram[51])

    # ball off-screen between points -> NaN the ball fields
    if bx == 0 and by == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan, lp, rp, 0.0], dtype=np.float32), None

    if prev_ball is None:
        vx, vy = np.nan, np.nan
    else:
        vx = float(bx - prev_ball[0])
        vy = float(by - prev_ball[1])

    collision = 0.0
    if not np.isnan(vx):
        # right paddle x in Pong is around 140; ball approaching from the left at vx>0
        if vx > 0 and bx > (140 - COLLISION_X_THRESH) and bx < 140:
            collision = 1.0

    return np.array([bx, by, vx, vy, lp, rp, collision], dtype=np.float32), (bx, by)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env()
    rng = np.random.default_rng(SEED)

    frames_list, actions_list, labels_list = [], [], []
    total = 0
    ep = 0

    while total < TARGET_FRAMES:
        obs, _ = env.reset(seed=SEED + ep)
        prev_ball = None
        for _ in range(MAX_STEPS_PER_EP):
            action = int(rng.integers(0, env.action_space.n))
            obs, _, terminated, truncated, _ = env.step(action)
            ram = env.unwrapped.ale.getRAM()
            lbl, prev_ball = read_ram_labels(ram, prev_ball)

            frames_list.append(obs)
            actions_list.append(action)
            labels_list.append(lbl)
            total += 1

            if total % 2000 == 0:
                valid = sum(1 for l in labels_list if not np.isnan(l[0]))
                print(f"  collected {total} / {TARGET_FRAMES}  ({valid} ball-visible)")
            if terminated or truncated or total >= TARGET_FRAMES:
                break
        ep += 1

    env.close()

    frames = np.stack(frames_list, axis=0).astype(np.uint8)
    actions = np.array(actions_list, dtype=np.int64)
    labels = np.stack(labels_list, axis=0).astype(np.float32)

    assert frames.shape[1:] == (210, 160, 3), f"unexpected frame shape {frames.shape}"
    assert labels.shape[1] == 7

    print(f"\nfinal: frames {frames.shape}  actions {actions.shape}  labels {labels.shape}")
    print(f"episodes: {ep}")
    print(f"ball-visible frames: {int((~np.isnan(labels[:,0])).sum())} / {len(labels)}")
    print(f"collision_imminent frames: {int(labels[:,6].sum())}")

    np.savez_compressed(
        OUT_PATH,
        frames=frames,
        actions=actions,
        labels=labels,
        label_keys=LABEL_KEYS,
    )
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"saved {OUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
