#!/usr/bin/env python3
"""
evaluate.py — Multi-Algorithm Evaluation & Comparison
======================================================
Usage:
    python3 evaluate.py
    python3 evaluate.py --algo td3 sac --episodes 50
"""
import argparse, json, os
import numpy as np
from stable_baselines3 import DQN, DDPG, TD3, SAC
from nav_env import NavigationEnv
from config import EnvConfig, MODEL_DIR, EVAL_DIR, ALGORITHMS

ALGO_CLASSES = {"dqn": DQN, "ddpg": DDPG, "td3": TD3, "sac": SAC}

def evaluate_model(algo_name, n_episodes=20, env_config=None, deterministic=True, verbose=True):
    if env_config is None: env_config = EnvConfig()
    model_path = os.path.join(MODEL_DIR, algo_name, f"{algo_name}_final")
    if not os.path.exists(model_path + ".zip"):
        print(f"  No model for {algo_name.upper()}"); return None
    model = ALGO_CLASSES[algo_name].load(model_path)
    is_discrete = ALGORITHMS[algo_name]["discrete"]
    env = NavigationEnv(discrete_action=is_discrete, max_episode_steps=env_config.max_episode_steps,
                        sensor_timeout=env_config.sensor_timeout)
    rewards, lengths, successes, collisions, dists = [], [], [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(); done, ep_r, ep_l = False, 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(action)
            ep_r += r; ep_l += 1; done = term or trunc
        rewards.append(ep_r); lengths.append(ep_l)
        successes.append(info.get("goal_reached", False))
        collisions.append(info.get("collision", False))
        dists.append(info.get("goal_distance", -1))
        if verbose:
            s = "GOAL" if info.get("goal_reached") else ("CRASH" if info.get("collision") else "TIMEOUT")
            print(f"    Ep {ep+1:>3d}: reward={ep_r:+8.1f} len={ep_l:>4d} {s}")
    env.close()
    return {"algorithm": algo_name.upper(), "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)), "success_rate": float(np.mean(successes)),
            "collision_rate": float(np.mean(collisions)), "avg_length": float(np.mean(lengths)),
            "avg_final_dist": float(np.mean(dists))}

def compare_algorithms(algos, n_episodes=20):
    print("=" * 60 + "\n  Multi-Algorithm Comparison\n" + "=" * 60)
    results = {}
    for a in algos:
        r = evaluate_model(a, n_episodes); 
        if r: results[a] = r
    print(f"\n{'Algo':<8} {'Reward':>10} {'Success':>9} {'Collision':>11} {'Length':>8}")
    print("-" * 50)
    for a, r in sorted(results.items(), key=lambda x: x[1]["avg_reward"], reverse=True):
        print(f"{a.upper():<8} {r['avg_reward']:>+9.1f} {r['success_rate']:>8.1%} {r['collision_rate']:>10.1%} {r['avg_length']:>7.0f}")
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(os.path.join(EVAL_DIR, "comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", nargs="+", default=["dqn","ddpg","td3","sac"])
    p.add_argument("--episodes", type=int, default=20)
    a = p.parse_args()
    compare_algorithms(a.algo, a.episodes)
