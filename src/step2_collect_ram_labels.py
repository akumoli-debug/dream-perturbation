import os, pickle
import numpy as np
import gymnasium as gym
import ale_py
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation

gym.register_envs(ale_py)

def make_env(game):
    env = gym.make(f"ALE/{game}-v5", obs_type="rgb", frameskip=4, repeat_action_probability=0.0)
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=(84, 84))
    return env

def get_ball(game, ram):
    if game == "Pong":
        return int(ram[49]), int(ram[54])
    return int(ram[99]), int(ram[101])

def collect(game, n_episodes=200, max_steps=1000):
    env = make_env(game)
    frames, labels = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            ram = env.unwrapped.ale.getRAM()
            bx, by = get_ball(game, ram)
            if not (bx == 0 and by == 0):
                frames.append(obs[np.newaxis])
                labels.append([bx, by])
            if terminated or truncated:
                break
        if ep % 20 == 0:
            print(f"[{game}] ep {ep}/{n_episodes} — {len(frames)} samples")
    env.close()
    return {"frames": np.stack(frames).astype(np.uint8), "labels": np.array(labels, dtype=np.float32)}

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    for game in ["Pong", "Breakout"]:
        print(f"\n=== {game} ===")
        data = collect(game)
        with open(f"data/ram_labels_{game.lower()}.pkl", "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {len(data['frames'])} samples")
