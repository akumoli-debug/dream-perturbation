"""
Fix 1 — Consecutive Frame Context
===================================
PROBLEM:
  All probe training and fine-tuning used:
    obs = batch.unsqueeze(1).expand(-1, 4, -1, -1, -1).reshape(B, 12, 84, 84)
  This passes the same single frame 4 times as temporal context.
  DIAMOND was trained on 4 DISTINCT consecutive frames.
  Every R² result is potentially an artifact of this OOD input.

FIX:
  Step 1: Re-collect data as (context_4frames, ball_y_label) pairs
          where context_4frames = frames[t-3], frames[t-2], frames[t-1], frames[t]
          and ball_y_label = RAM position at frame t
  Step 2: Save to data/context_labels_pong.pkl
  Step 3: Use this in probe training and fine-tuning

HOW TO RUN:
  python3 src/fix1_consecutive_frames.py
  # Takes ~5 minutes, same as original collection
  # Output: data/context_labels_pong.pkl
"""

import os, sys, pickle
import numpy as np
import gymnasium as gym
import ale_py
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation

sys.path.insert(0, "./src")
gym.register_envs(ale_py)

CONTEXT_LEN = 4   # DIAMOND uses 4 frames of context

def make_env(game):
    env = gym.make(f"ALE/{game}-v5", obs_type="rgb",
                   frameskip=4, repeat_action_probability=0.0)
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=(84, 84))
    return env

def get_ball(game, ram):
    if game == "Pong":
        return int(ram[49]), int(ram[54])
    return int(ram[99]), int(ram[101])

def collect_with_context(game, n_episodes=200, max_steps=1000):
    """
    Returns:
      contexts: [N, 4, 84, 84] uint8  — 4 consecutive frames ending at label frame
      labels:   [N, 2] float32         — (ball_x, ball_y) at the final frame
    """
    env = make_env(game)
    contexts, labels = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        # Ring buffer of last CONTEXT_LEN frames
        frame_buffer = [obs.copy() for _ in range(CONTEXT_LEN)]

        for step in range(max_steps):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)

            # Update ring buffer
            frame_buffer.pop(0)
            frame_buffer.append(obs.copy())

            # Get ball position from RAM
            ram = env.unwrapped.ale.getRAM()
            bx, by = get_ball(game, ram)

            # Skip off-screen frames
            if bx == 0 and by == 0:
                if terminated or truncated:
                    break
                continue

            # Stack 4 consecutive frames as context
            context = np.stack(frame_buffer, axis=0)  # [4, 84, 84]
            contexts.append(context)
            labels.append([bx, by])

            if terminated or truncated:
                break

        if ep % 20 == 0:
            print(f"[{game}] ep {ep}/{n_episodes} — {len(contexts)} samples")

    env.close()
    return {
        "contexts": np.array(contexts, dtype=np.uint8),   # [N, 4, 84, 84]
        "labels":   np.array(labels,   dtype=np.float32), # [N, 2]
    }

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    for game in ["Pong", "Breakout"]:
        print(f"\n=== Collecting {game} with consecutive context ===")
        data = collect_with_context(game, n_episodes=200)
        path = f"data/context_labels_{game.lower()}.pkl"
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {len(data['contexts'])} samples → {path}")
        print(f"Context shape: {data['contexts'].shape}")
        print(f"Labels shape:  {data['labels'].shape}")
