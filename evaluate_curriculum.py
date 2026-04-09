#!/usr/bin/env python3
"""
evaluate_curriculum.py — Compare DQN, DDPG, TD3 across all 4 curriculum phases
================================================================================
Loads each algorithm's Phase 4 (final) model and runs N evaluation episodes.
Also loads per-phase training metadata to compare convergence speed.

Usage:
    # Evaluate final models (Phase 4) on the live simulation:
    python3 evaluate_curriculum.py --episodes 20

    # Compare training metrics only (no sim needed):
    python3 evaluate_curriculum.py --metrics-only
"""

import argparse
import json
import os
from datetime import timedelta

import numpy as np

from stable_baselines3 import DQN, DDPG, TD3, SAC
from nav_env import NavigationEnv
from config import EnvConfig, MODEL_DIR, EVAL_DIR, ALGORITHMS

ALGO_CLASSES = {"dqn": DQN, "ddpg": DDPG, "td3": TD3, "sac": SAC}
ALGOS_TO_EVAL = ["td3", "ddpg", "dqn"]  # order: expected best → worst
PHASES = [1, 2, 3, 4]


def load_phase_metrics(algo_name: str) -> dict:
    """Load saved training metrics for all phases of an algorithm."""
    phase_data = {}
    for phase in PHASES:
        filepath = os.path.join(EVAL_DIR, f"{algo_name}_phase{phase}_results.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                phase_data[phase] = json.load(f)
    return phase_data


def evaluate_model(algo_name: str, phase: int = 4, n_episodes: int = 20,
                   deterministic: bool = True, verbose: bool = True) -> dict:
    """Evaluate an algorithm's model from a specific phase."""
    is_discrete = ALGORITHMS[algo_name]["discrete"]
    model_path = os.path.join(MODEL_DIR, algo_name, f"{algo_name}_phase{phase}")

    if not os.path.exists(model_path + ".zip"):
        print(f"  ❌  No Phase {phase} model for {algo_name.upper()} at {model_path}")
        return None

    print(f"\n  🔍  Evaluating {algo_name.upper()} Phase {phase} — {n_episodes} episodes")
    model = ALGO_CLASSES[algo_name].load(model_path)

    env_cfg = EnvConfig()
    env = NavigationEnv(
        discrete_action=is_discrete,
        max_episode_steps=env_cfg.max_episode_steps,
        sensor_timeout=env_cfg.sensor_timeout,
    )

    rewards, lengths, successes, collisions, final_dists = [], [], [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            ep_len += 1
            done = term or trunc

        rewards.append(ep_reward)
        lengths.append(ep_len)
        successes.append(info.get("goal_reached", False))
        collisions.append(info.get("collision", False))
        final_dists.append(info.get("goal_distance", -1))

        if verbose:
            outcome = ("✅ GOAL" if info.get("goal_reached")
                       else ("💥 CRASH" if info.get("collision") else "⏰ TIMEOUT"))
            print(f"    Ep {ep+1:>3d}: reward={ep_reward:+8.1f}  "
                  f"len={ep_len:>4d}  {outcome}")

    env.close()

    return {
        "algorithm": algo_name.upper(),
        "phase": phase,
        "n_episodes": n_episodes,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "max_reward": float(np.max(rewards)),
        "min_reward": float(np.min(rewards)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "timeout_rate": float(1.0 - np.mean(successes) - np.mean(collisions)),
        "avg_episode_length": float(np.mean(lengths)),
        "avg_final_goal_dist": float(np.mean([d for d in final_dists if d >= 0])),
    }


def print_training_comparison(algos: list):
    """Print convergence speed comparison from saved training metadata."""
    print("\n" + "=" * 80)
    print("  📊  TRAINING METRICS COMPARISON (Convergence Speed)")
    print("=" * 80)

    header = f"{'Algo':<8} {'Phase':<20} {'Steps':>8} {'Time':>12}"
    print(f"\n{header}")
    print("-" * 55)

    total_times = {}
    total_steps = {}

    for algo in algos:
        phase_data = load_phase_metrics(algo)
        total_times[algo] = 0
        total_steps[algo] = 0

        if not phase_data:
            print(f"{algo.upper():<8}  — no training data found —")
            continue

        for phase in PHASES:
            if phase in phase_data:
                d = phase_data[phase]
                steps = d.get("total_timesteps", 0)
                secs = d.get("training_time_seconds", 0)
                total_times[algo] += secs
                total_steps[algo] += steps
                time_str = str(timedelta(seconds=int(secs)))
                label = d.get("phase_label", f"Phase {phase}")
                print(f"{algo.upper():<8} {label:<20} {steps:>8,} {time_str:>12}")

        # Total row
        total_t = str(timedelta(seconds=int(total_times[algo])))
        print(f"{algo.upper():<8} {'TOTAL':<20} {total_steps[algo]:>8,} {total_t:>12}")
        print()

    # ── Summary table ─────────────────────────────────────────────────
    if any(total_times.values()):
        print("\n" + "=" * 80)
        print("  ⏱️  CONVERGENCE SPEED RANKING")
        print("=" * 80)
        print(f"\n{'Rank':<6} {'Algo':<8} {'Total Steps':>12} {'Total Time':>14} {'Steps/sec':>10}")
        print("-" * 55)
        ranked = sorted(
            [(a, total_steps[a], total_times[a]) for a in algos if total_times[a] > 0],
            key=lambda x: x[2]  # sort by total wall-clock time
        )
        for rank, (algo, steps, secs) in enumerate(ranked, 1):
            time_str = str(timedelta(seconds=int(secs)))
            sps = steps / secs if secs > 0 else 0
            print(f"  {rank}    {algo.upper():<8} {steps:>12,} {time_str:>14} {sps:>9.1f}")


def print_evaluation_comparison(results: dict):
    """Print side-by-side evaluation table."""
    print("\n" + "=" * 80)
    print("  🏆  FINAL MODEL PERFORMANCE COMPARISON (Phase 4)")
    print("=" * 80)

    header = (f"\n{'Algo':<8} {'Avg Reward':>11} {'Success':>9} "
              f"{'Collision':>11} {'Timeout':>9} {'Avg Len':>8} {'Goal Dist':>10}")
    print(header)
    print("-" * 72)

    ranked = sorted(results.items(),
                    key=lambda x: x[1]["success_rate"], reverse=True)

    for algo, r in ranked:
        print(f"{algo.upper():<8} {r['avg_reward']:>+10.1f} "
              f"{r['success_rate']:>8.1%} "
              f"{r['collision_rate']:>10.1%} "
              f"{r['timeout_rate']:>8.1%} "
              f"{r['avg_episode_length']:>7.0f} "
              f"{r['avg_final_goal_dist']:>9.2f}m")

    # ── Winner announcement ───────────────────────────────────────────
    if ranked:
        winner = ranked[0]
        print(f"\n  🥇  WINNER: {winner[0].upper()}")
        print(f"      Success rate: {winner[1]['success_rate']:.1%}")
        print(f"      Collision rate: {winner[1]['collision_rate']:.1%}")
        print(f"      Avg reward: {winner[1]['avg_reward']:+.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare curriculum-trained models",
    )
    parser.add_argument("--algo", nargs="+", default=ALGOS_TO_EVAL,
                        help="Algorithms to evaluate")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes per algorithm")
    parser.add_argument("--phase", type=int, default=4,
                        help="Which phase model to evaluate (default: 4)")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Only compare training metrics (no live sim needed)")
    args = parser.parse_args()

    # ── Training metrics (always available) ───────────────────────────
    print_training_comparison(args.algo)

    # ── Live evaluation (requires Webots running) ─────────────────────
    if not args.metrics_only:
        results = {}
        for algo in args.algo:
            r = evaluate_model(algo, phase=args.phase, n_episodes=args.episodes)
            if r:
                results[algo] = r

        if results:
            print_evaluation_comparison(results)

            # Save comparison results
            os.makedirs(EVAL_DIR, exist_ok=True)
            outpath = os.path.join(EVAL_DIR, "curriculum_comparison.json")
            with open(outpath, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  💾  Results saved: {outpath}")
    else:
        print("\n  ℹ️  Skipped live evaluation (--metrics-only). "
              "Run without --metrics-only with Webots active to evaluate models.")


if __name__ == "__main__":
    main()
