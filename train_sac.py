#!/usr/bin/env python3
"""
train_sac.py — Modified Soft Actor-Critic Training Script
==========================================================
Trains a Modified SAC on NavigationEnv with continuous actions.

Modifications over vanilla SAC:
    1. Automatic entropy coefficient tuning (ent_coef="auto")
    2. Navigation-aware reward shaping (handled by env)
    3. Configurable target entropy for aggressive/conservative exploration
    4. Optional State-Dependent Exploration (SDE) for smoother actions

SAC advantages for robotic navigation:
    - Maximum entropy framework encourages exploration
    - Stable training due to clipped double-Q (like TD3)
    - Stochastic policy naturally handles multi-modal action distributions
    - Sample-efficient off-policy learning

Usage:
    python3 train_sac.py [--timesteps 50000] [--seed 42] [--use-sde]
"""

import argparse
import os

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from config import SACConfig, EnvConfig, LOG_DIR, MODEL_DIR
from utils import (
    make_env,
    NavigationMetricsCallback,
    CheckpointCallback,
    TrainingTimer,
    save_training_results,
)

ALGO_NAME = "sac"


def train_sac(cfg: SACConfig = None, env_cfg: EnvConfig = None):
    """Train Modified SAC agent."""
    if cfg is None:
        cfg = SACConfig()
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

    # ── Policy kwargs ─────────────────────────────────────────────────
    policy_kwargs = {
        "net_arch": cfg.net_arch,
    }

    # ── Create SAC model ──────────────────────────────────────────────
    # Key modification: automatic entropy tuning with configurable target
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        learning_starts=cfg.learning_starts,
        gamma=cfg.gamma,
        tau=cfg.tau,
        ent_coef=cfg.ent_coef,              # "auto" = learned entropy coeff
        target_entropy=cfg.target_entropy,   # "auto" = -dim(action_space)
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        use_sde=cfg.use_sde,                # state-dependent exploration
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_path,
        seed=cfg.seed,
        verbose=1,
    )

    print(f"  Model: {model.policy}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Entropy coeff: {cfg.ent_coef}")
    print(f"  Target entropy: {cfg.target_entropy}")
    print(f"  Use SDE: {cfg.use_sde}")
    print(f"  Total timesteps: {cfg.total_timesteps:,}")

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = [
        NavigationMetricsCallback(
            eval_freq=2_000,
            log_prefix="sac",
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=10_000,
            save_path=model_path,
            algo_name=ALGO_NAME,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────
    with TrainingTimer(f"Modified {ALGO_NAME.upper()}") as timer:
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

    # ── Log SAC-specific learned parameters ───────────────────────────
    try:
        learned_ent_coef = model.ent_coef_tensor.item()
        print(f"  🌡️   Learned entropy coeff: {learned_ent_coef:.4f}")
    except Exception:
        learned_ent_coef = "N/A"

    # ── Save training metadata ────────────────────────────────────────
    save_training_results(ALGO_NAME, {
        "algorithm": f"Modified {ALGO_NAME.upper()}",
        "total_timesteps": cfg.total_timesteps,
        "training_time_seconds": timer.elapsed,
        "seed": cfg.seed,
        "learning_rate": cfg.learning_rate,
        "buffer_size": cfg.buffer_size,
        "batch_size": cfg.batch_size,
        "ent_coef_initial": cfg.ent_coef,
        "ent_coef_learned": str(learned_ent_coef),
        "target_entropy": cfg.target_entropy,
        "use_sde": cfg.use_sde,
        "net_arch": cfg.net_arch,
        "model_path": final_path,
        "modifications": [
            "Automatic entropy coefficient tuning",
            "Navigation-aware dense reward shaping",
            "Frame-stacked observations for dynamic obstacle inference",
        ],
    })

    env.close()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Modified SAC")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--use-sde", action="store_true",
                        help="Enable State-Dependent Exploration")
    args = parser.parse_args()

    cfg = SACConfig()
    if args.timesteps:
        cfg.total_timesteps = args.timesteps
    if args.seed:
        cfg.seed = args.seed
    if args.lr:
        cfg.learning_rate = args.lr
    if args.use_sde:
        cfg.use_sde = True

    train_sac(cfg)


if __name__ == "__main__":
    main()
