#!/usr/bin/env python3
"""
train_ddpg.py — DDPG Training Script
======================================
Trains Deep Deterministic Policy Gradient on NavigationEnv
with continuous action space Box(2,).

Usage:
    python3 train_ddpg.py [--timesteps 50000] [--seed 42]
"""

import argparse
import os

import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from config import DDPGConfig, EnvConfig, LOG_DIR, MODEL_DIR
from utils import (
    make_env,
    NavigationMetricsCallback,
    CheckpointCallback,
    TrainingTimer,
    save_training_results,
)

ALGO_NAME = "ddpg"


def train_ddpg(cfg: DDPGConfig = None, env_cfg: EnvConfig = None):
    """Train DDPG agent."""
    if cfg is None:
        cfg = DDPGConfig()
    if env_cfg is None:
        env_cfg = EnvConfig()

    log_path = os.path.join(LOG_DIR, ALGO_NAME)
    model_path = os.path.join(MODEL_DIR, ALGO_NAME)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # ── Create environment (continuous actions) ───────────────────────
    env = DummyVecEnv([make_env(
        discrete=False,
        env_config=env_cfg,
        log_dir=log_path,
    )])

    # ── Action noise ──────────────────────────────────────────────────
    n_actions = env.action_space.shape[-1]  # 2

    if cfg.noise_type == "ornstein-uhlenbeck":
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=cfg.noise_std * np.ones(n_actions),
        )
    else:
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=cfg.noise_std * np.ones(n_actions),
        )

    # ── Create DDPG model ─────────────────────────────────────────────
    model = DDPG(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        learning_starts=cfg.learning_starts,
        gamma=cfg.gamma,
        tau=cfg.tau,
        train_freq=cfg.train_freq,
        action_noise=action_noise,
        policy_kwargs={"net_arch": cfg.net_arch},
        tensorboard_log=log_path,
        seed=cfg.seed,
        verbose=1,
    )

    print(f"  Model: {model.policy}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Action noise: {cfg.noise_type} (σ={cfg.noise_std})")
    print(f"  Total timesteps: {cfg.total_timesteps:,}")

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = [
        NavigationMetricsCallback(
            eval_freq=2_000,
            log_prefix="ddpg",
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=10_000,
            save_path=model_path,
            algo_name=ALGO_NAME,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────
    with TrainingTimer(ALGO_NAME.upper()) as timer:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callbacks,
            tb_log_name=ALGO_NAME,
            progress_bar=True,
        )

    # ── Save final model ──────────────────────────────────────────────
    final_path = os.path.join(model_path, f"{ALGO_NAME}_final")
    model.save(final_path)
    print(f"  💾  Final model saved: {final_path}")

    # ── Save training metadata ────────────────────────────────────────
    save_training_results(ALGO_NAME, {
        "algorithm": ALGO_NAME.upper(),
        "total_timesteps": cfg.total_timesteps,
        "training_time_seconds": timer.elapsed,
        "seed": cfg.seed,
        "learning_rate": cfg.learning_rate,
        "buffer_size": cfg.buffer_size,
        "batch_size": cfg.batch_size,
        "noise_type": cfg.noise_type,
        "noise_std": cfg.noise_std,
        "net_arch": cfg.net_arch,
        "model_path": final_path,
    })

    env.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train DDPG")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = DDPGConfig()
    if args.timesteps:
        cfg.total_timesteps = args.timesteps
    if args.seed:
        cfg.seed = args.seed
    if args.lr:
        cfg.learning_rate = args.lr

    train_ddpg(cfg)


if __name__ == "__main__":
    main()
