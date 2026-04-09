#!/usr/bin/env python3
"""
utils.py — Training Utilities
==============================
Shared utilities for logging, evaluation callbacks, and checkpointing.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from nav_env import NavigationEnv
from config import EnvConfig, LOG_DIR, MODEL_DIR, EVAL_DIR


# ═══════════════════════════════════════════════════════════════════════════════
# Environment factory
# ═══════════════════════════════════════════════════════════════════════════════

def make_env(
    discrete: bool = False,
    env_config: Optional[EnvConfig] = None,
    log_dir: Optional[str] = None,
    rank: int = 0,
):
    """
    Create a wrapped NavigationEnv instance.

    Returns a function (for DummyVecEnv compatibility) that creates:
        NavigationEnv → Monitor
    """
    if env_config is None:
        env_config = EnvConfig()

    def _init():
        env = NavigationEnv(
            discrete_action=discrete,
            goal_position=env_config.goal_position,
            max_episode_steps=env_config.max_episode_steps,
            collision_threshold=env_config.collision_threshold,
            goal_threshold=env_config.goal_threshold,
            v_max=env_config.v_max,
            w_max=env_config.w_max,
            max_lidar_range=env_config.max_lidar_range,
            max_goal_distance=env_config.max_goal_distance,
            sensor_timeout=env_config.sensor_timeout,
            num_lidar_bins=env_config.num_lidar_bins,
            num_frames=env_config.num_frames,
        )

        # Wrap with Monitor for episode logging
        monitor_path = None
        if log_dir is not None:
            monitor_path = os.path.join(log_dir, f"monitor_{rank}")
            os.makedirs(os.path.dirname(monitor_path) if os.path.dirname(monitor_path) else ".", exist_ok=True)

        env = Monitor(env, filename=monitor_path)
        return env

    return _init


# ═══════════════════════════════════════════════════════════════════════════════
# Custom Callbacks
# ═══════════════════════════════════════════════════════════════════════════════

class NavigationMetricsCallback(BaseCallback):
    """
    Custom callback that logs navigation-specific metrics to TensorBoard:
        - Success rate (goal reached)
        - Collision rate
        - Average goal distance at episode end
        - Average episode reward
        - Average episode length
    """

    def __init__(
        self,
        eval_freq: int = 1_000,
        n_eval_episodes: int = 10,
        log_prefix: str = "nav",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_prefix = log_prefix

        # Episode tracking
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []
        self._episode_successes: List[bool] = []
        self._episode_collisions: List[bool] = []
        self._episode_final_dists: List[float] = []
        self._current_episode_reward: float = 0.0

    def _on_step(self) -> bool:
        # Accumulate reward
        if self.locals.get("rewards") is not None:
            self._current_episode_reward += self.locals["rewards"][0]

        # Check for episode end
        if self.locals.get("dones") is not None and self.locals["dones"][0]:
            infos = self.locals.get("infos", [{}])
            info = infos[0] if infos else {}

            self._episode_rewards.append(self._current_episode_reward)
            self._episode_lengths.append(info.get("step", 0))
            self._episode_successes.append(info.get("goal_reached", False))
            self._episode_collisions.append(info.get("collision", False))
            self._episode_final_dists.append(info.get("goal_distance", -1))

            self._current_episode_reward = 0.0

        # Log metrics periodically
        if self.n_calls % self.eval_freq == 0 and len(self._episode_rewards) > 0:
            n = min(len(self._episode_rewards), 100)  # last 100 episodes
            recent_rewards = self._episode_rewards[-n:]
            recent_lengths = self._episode_lengths[-n:]
            recent_successes = self._episode_successes[-n:]
            recent_collisions = self._episode_collisions[-n:]
            recent_dists = self._episode_final_dists[-n:]

            success_rate = sum(recent_successes) / len(recent_successes)
            collision_rate = sum(recent_collisions) / len(recent_collisions)
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            avg_final_dist = np.mean([d for d in recent_dists if d >= 0])

            self.logger.record(f"{self.log_prefix}/success_rate", success_rate)
            self.logger.record(f"{self.log_prefix}/collision_rate", collision_rate)
            self.logger.record(f"{self.log_prefix}/avg_reward", avg_reward)
            self.logger.record(f"{self.log_prefix}/avg_episode_length", avg_length)
            self.logger.record(f"{self.log_prefix}/avg_final_goal_dist", avg_final_dist)
            self.logger.record(f"{self.log_prefix}/total_episodes", len(self._episode_rewards))

            if self.verbose >= 1:
                print(
                    f"  [{self.num_timesteps:>7,} steps]  "
                    f"reward={avg_reward:+.1f}  "
                    f"success={success_rate:.1%}  "
                    f"collision={collision_rate:.1%}  "
                    f"len={avg_length:.0f}  "
                    f"episodes={len(self._episode_rewards)}"
                )

        return True


class CheckpointCallback(BaseCallback):
    """Save model checkpoints at regular intervals.
    Validates model outputs before saving to prevent persisting corrupted weights.
    """

    def __init__(
        self,
        save_freq: int = 10_000,
        save_path: str = MODEL_DIR,
        algo_name: str = "model",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.algo_name = algo_name

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Validate model before saving — don't persist NaN-corrupted weights
            try:
                test_obs = self.training_env.observation_space.sample()
                test_action, _ = self.model.predict(test_obs, deterministic=True)
                if not np.all(np.isfinite(test_action)):
                    print(f"  ⚠️  Skipping checkpoint at {self.num_timesteps} — "
                          f"model outputs NaN (weights corrupted)")
                    return True  # continue training, just don't save
            except Exception:
                pass  # if validation itself fails, save anyway as backup

            path = os.path.join(
                self.save_path,
                f"{self.algo_name}_{self.num_timesteps}_steps"
            )
            self.model.save(path)
            if self.verbose >= 1:
                print(f"  💾  Checkpoint saved: {path}")
        return True


class SuccessRateStopCallback(BaseCallback):
    """
    Early stopping callback: halts training when success rate >= threshold.
    Monitors episode outcomes and checks every `check_freq` steps.
    """

    def __init__(
        self,
        success_threshold: float = 0.90,
        min_episodes: int = 50,
        check_freq: int = 2_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.success_threshold = success_threshold
        self.min_episodes = min_episodes
        self.check_freq = check_freq
        self._successes: List[bool] = []

    def _on_step(self) -> bool:
        # Track episode outcomes
        if self.locals.get("dones") is not None and self.locals["dones"][0]:
            infos = self.locals.get("infos", [{}])
            info = infos[0] if infos else {}
            self._successes.append(info.get("goal_reached", False))

        # Check periodically
        if self.n_calls % self.check_freq == 0:
            if len(self._successes) >= self.min_episodes:
                recent = self._successes[-self.min_episodes:]
                rate = sum(recent) / len(recent)
                if rate >= self.success_threshold:
                    if self.verbose >= 1:
                        print(f"\n  🎯  SUCCESS RATE {rate:.1%} >= {self.success_threshold:.0%} "
                              f"over last {self.min_episodes} episodes!")
                        print(f"  🛑  Early stopping at {self.num_timesteps:,} steps.\n")
                    return False  # Stop training
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Training timer
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingTimer:
    """Simple context manager for timing training runs."""

    def __init__(self, algo_name: str):
        self.algo_name = algo_name
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        print(f"\n{'═' * 60}")
        print(f"  🚀  Starting {self.algo_name} training")
        print(f"{'═' * 60}\n")
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        minutes = self.elapsed / 60
        print(f"\n{'═' * 60}")
        print(f"  ✅  {self.algo_name} training complete!")
        print(f"  ⏱   Duration: {minutes:.1f} minutes ({self.elapsed:.0f}s)")
        print(f"{'═' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Results I/O
# ═══════════════════════════════════════════════════════════════════════════════

def save_training_results(
    algo_name: str,
    metrics: Dict[str, Any],
    filepath: Optional[str] = None,
):
    """Save training results as JSON."""
    if filepath is None:
        filepath = os.path.join(EVAL_DIR, f"{algo_name}_results.json")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"  📊  Results saved: {filepath}")


def load_training_results(algo_name: str) -> Dict[str, Any]:
    """Load training results from JSON."""
    filepath = os.path.join(EVAL_DIR, f"{algo_name}_results.json")
    with open(filepath, "r") as f:
        return json.load(f)
