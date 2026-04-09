#!/usr/bin/env python3
"""
config.py — Shared Hyperparameters & Experiment Configuration
=============================================================
Central configuration for all DRL training scripts.
Modify values here to affect DQN, DDPG, TD3, and SAC simultaneously.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Store training outputs on D: drive to save C: drive space
# In WSL, D:\ is accessible as /mnt/d/
_OUTPUT_ROOT = os.path.join("/mnt/d", "H2F_training")
LOG_DIR = os.path.join(_OUTPUT_ROOT, "logs")
MODEL_DIR = os.path.join(_OUTPUT_ROOT, "models")
EVAL_DIR = os.path.join(_OUTPUT_ROOT, "eval_results")

# Create directories
for d in [LOG_DIR, MODEL_DIR, EVAL_DIR]:
    os.makedirs(d, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Environment Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EnvConfig:
    """Configuration for NavigationEnv."""
    goal_position: Optional[Tuple[float, float]] = None  # None = random goals
    max_episode_steps: int = 200
    collision_threshold: float = 0.20       # metres
    goal_threshold: float = 0.35            # metres
    v_max: float = 0.22                     # m/s
    w_max: float = 2.0                      # rad/s
    max_lidar_range: float = 3.5            # metres
    max_goal_distance: float = 10.0         # metres
    sensor_timeout: float = 5.0             # seconds
    num_lidar_bins: int = 24
    num_frames: int = 3                     # frame stacking depth


# ═══════════════════════════════════════════════════════════════════════════════
# Training Hyperparameters
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DQNConfig:
    """DQN-specific hyperparameters."""
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    batch_size: int = 64
    learning_starts: int = 1_000
    gamma: float = 0.99
    tau: float = 1.0                        # hard update
    target_update_interval: int = 1_000
    train_freq: int = 4
    exploration_fraction: float = 0.3
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    seed: int = 42


@dataclass
class DDPGConfig:
    """DDPG-specific hyperparameters."""
    total_timesteps: int = 50_000
    learning_rate: float = 1e-3
    buffer_size: int = 100_000
    batch_size: int = 100
    learning_starts: int = 1_000
    gamma: float = 0.99
    tau: float = 0.005                      # soft update
    train_freq: Tuple[int, str] = (1, "episode")
    noise_type: str = "ornstein-uhlenbeck"  # or "normal"
    noise_std: float = 0.1
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    seed: int = 42


@dataclass
class TD3Config:
    """TD3-specific hyperparameters."""
    total_timesteps: int = 50_000
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    batch_size: int = 128
    learning_starts: int = 500
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2                   # TD3 hallmark: delayed updates
    target_policy_noise: float = 0.2        # smoothing noise
    target_noise_clip: float = 0.5
    train_freq: Tuple[int, str] = (4, "step")
    noise_type: str = "normal"
    noise_std: float = 0.1
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    seed: int = 42


@dataclass
class SACConfig:
    """Modified SAC hyperparameters."""
    total_timesteps: int = 50_000
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    batch_size: int = 256
    learning_starts: int = 1_000
    gamma: float = 0.99
    tau: float = 0.005
    ent_coef: str = "auto"                  # automatic entropy tuning
    target_entropy: str = "auto"            # = -dim(action_space)
    train_freq: int = 1
    gradient_steps: int = 1
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    use_sde: bool = False                   # state-dependent exploration
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalConfig:
    """Evaluation parameters."""
    n_eval_episodes: int = 20
    eval_freq: int = 5_000                  # evaluate every N training steps
    deterministic: bool = True
    n_seeds: int = 3                        # number of random seeds for comparison
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


# ═══════════════════════════════════════════════════════════════════════════════
# Algorithm registry
# ═══════════════════════════════════════════════════════════════════════════════

ALGORITHMS = {
    "dqn": {
        "discrete": True,
        "config_class": DQNConfig,
        "description": "Deep Q-Network (discrete actions)",
    },
    "ddpg": {
        "discrete": False,
        "config_class": DDPGConfig,
        "description": "Deep Deterministic Policy Gradient (continuous)",
    },
    "td3": {
        "discrete": False,
        "config_class": TD3Config,
        "description": "Twin Delayed DDPG (continuous)",
    },
    "sac": {
        "discrete": False,
        "config_class": SACConfig,
        "description": "Soft Actor-Critic (continuous, auto-entropy)",
    },
}


if __name__ == "__main__":
    print("Algorithm Configurations:")
    print("=" * 60)
    for name, info in ALGORITHMS.items():
        cfg = info["config_class"]()
        print(f"\n{name.upper()} — {info['description']}")
        print(f"  Discrete: {info['discrete']}")
        print(f"  Timesteps: {cfg.total_timesteps:,}")
        print(f"  LR: {cfg.learning_rate}")
        print(f"  Batch: {cfg.batch_size}")
        print(f"  Net: {cfg.net_arch}")
