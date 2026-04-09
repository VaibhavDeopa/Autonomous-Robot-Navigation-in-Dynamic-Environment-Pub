#!/usr/bin/env python3
"""
train_curriculum.py — 4-Phase Curriculum Training Script
========================================================
Trains a single DRL algorithm through 4 progressive phases:

  Phase 1: Empty square room          (learn goal-seeking)
  Phase 2: Static obstacles           (learn obstacle avoidance)
  Phase 3: Slow dynamic obstacles     (learn velocity prediction)
  Phase 4: Fast dynamic obstacles     (learn agility)

Between phases, the trained model weights are PRESERVED.
The reward structure is CONSTANT across all phases.

Usage:
    # Set the phase BEFORE launching Webots:
    export CURRICULUM_PHASE=1
    ros2 launch nav_env_pkg nav_launch.py

    # Then in a separate terminal:
    python3 train_curriculum.py --algo td3 --phase 1 --timesteps 30000
    python3 train_curriculum.py --algo td3 --phase 2 --timesteps 40000
    python3 train_curriculum.py --algo td3 --phase 3 --timesteps 50000
    python3 train_curriculum.py --algo td3 --phase 4 --timesteps 60000

Notes:
    - Phase 1 always trains from scratch.
    - Phases 2-4 automatically load saved model from the previous phase.
    - You MUST restart Webots with the correct CURRICULUM_PHASE env var
      between phases (the supervisor reads this at startup).
"""

import argparse
import os
import sys

import numpy as np

from stable_baselines3 import DQN, DDPG, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from config import (
    DQNConfig, DDPGConfig, TD3Config, SACConfig,
    EnvConfig, LOG_DIR, MODEL_DIR,
)
from utils import (
    make_env,
    NavigationMetricsCallback,
    CheckpointCallback,
    SuccessRateStopCallback,
    TrainingTimer,
    save_training_results,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase-specific timestep limits (per algorithm)
# Training stops early if 90%+ success rate is reached
# ═══════════════════════════════════════════════════════════════════════════════
PHASE_DEFAULTS = {
    1: {"label": "Empty Room",       "timesteps": {"td3": 20_000, "ddpg": 20_000, "dqn": 30_000, "sac": 20_000}},
    2: {"label": "Static Obstacles", "timesteps": {"td3": 50_000, "ddpg": 50_000, "dqn": 60_000, "sac": 50_000}},
    3: {"label": "Slow Dynamic",     "timesteps": {"td3": 40_000, "ddpg": 40_000, "dqn": 50_000, "sac": 40_000}},
    4: {"label": "Fast Dynamic",     "timesteps": {"td3": 50_000, "ddpg": 50_000, "dqn": 60_000, "sac": 50_000}},
}

ALGO_MAP = {
    "dqn":  (DQN,  DQNConfig,  True),
    "ddpg": (DDPG, DDPGConfig, False),
    "td3":  (TD3,  TD3Config,  False),
    "sac":  (SAC,  SACConfig,  False),
}


def get_model_path(algo_name: str, phase: int) -> str:
    """Path to save/load model for a given algorithm and phase."""
    return os.path.join(MODEL_DIR, f"{algo_name}_phase{phase}")


def create_fresh_model(algo_name, env, cfg):
    """Create a new model from scratch (Phase 1)."""
    AlgoClass, _, is_discrete = ALGO_MAP[algo_name]

    common_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": cfg.learning_rate,
        "buffer_size": cfg.buffer_size,
        "batch_size": cfg.batch_size,
        "learning_starts": cfg.learning_starts,
        "gamma": cfg.gamma,
        "tau": cfg.tau,
        "policy_kwargs": {"net_arch": cfg.net_arch},
        "tensorboard_log": os.path.join(LOG_DIR, algo_name),
        "seed": cfg.seed,
        "verbose": 1,
    }

    if algo_name == "dqn":
        common_kwargs.update({
            "target_update_interval": cfg.target_update_interval,
            "train_freq": cfg.train_freq,
            "exploration_fraction": cfg.exploration_fraction,
            "exploration_initial_eps": cfg.exploration_initial_eps,
            "exploration_final_eps": cfg.exploration_final_eps,
        })
    elif algo_name == "ddpg":
        n_actions = env.action_space.shape[-1]
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
        common_kwargs["action_noise"] = action_noise
        common_kwargs["train_freq"] = cfg.train_freq
    elif algo_name == "td3":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=cfg.noise_std * np.ones(n_actions),
        )
        common_kwargs.update({
            "action_noise": action_noise,
            "policy_delay": cfg.policy_delay,
            "target_policy_noise": cfg.target_policy_noise,
            "target_noise_clip": cfg.target_noise_clip,
            "train_freq": cfg.train_freq,
        })
    elif algo_name == "sac":
        common_kwargs.update({
            "ent_coef": cfg.ent_coef,
            "target_entropy": cfg.target_entropy,
            "train_freq": cfg.train_freq,
            "gradient_steps": cfg.gradient_steps,
        })

    return AlgoClass(**common_kwargs)


def train_phase(algo_name: str, phase: int, timesteps: int, resume_from: str = None):
    """Train one algorithm for one curriculum phase."""
    AlgoClass, ConfigClass, is_discrete = ALGO_MAP[algo_name]
    cfg = ConfigClass()
    env_cfg = EnvConfig()

    phase_info = PHASE_DEFAULTS[phase]
    default_timesteps = phase_info["timesteps"].get(algo_name, 50_000)

    print("\n" + "=" * 70)
    print(f"  🎓  CURRICULUM TRAINING")
    print(f"  Algorithm : {algo_name.upper()}")
    print(f"  Phase     : {phase}/4 — {phase_info['label']}")
    print(f"  Timesteps : {timesteps:,} (max — stops early at 90% success)")
    print("=" * 70)

    # ── Verify CURRICULUM_PHASE env var matches ───────────────────────
    env_phase = int(os.environ.get("CURRICULUM_PHASE", "0"))
    if env_phase != phase:
        print(f"\n  ⚠️  WARNING: CURRICULUM_PHASE env var = {env_phase}, "
              f"but you requested phase {phase}.")
        print(f"  Make sure Webots was launched with: "
              f"export CURRICULUM_PHASE={phase}")
        print(f"  The supervisor reads this at startup to configure obstacles.\n")

    # ── Create environment ────────────────────────────────────────────
    log_path = os.path.join(LOG_DIR, f"{algo_name}_phase{phase}")
    model_dir = os.path.join(MODEL_DIR, algo_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = DummyVecEnv([make_env(
        discrete=is_discrete,
        env_config=env_cfg,
        log_dir=log_path,
    )])

    # Helper function to prevent critic instability on load
    def _lower_lr(m):
        finetune_lr = 1e-4
        m.learning_rate = finetune_lr
        if hasattr(m, "policy") and hasattr(m.policy, "optimizer"):
            m.policy.optimizer.param_groups[0]['lr'] = finetune_lr
        if hasattr(m, "actor") and hasattr(m.actor, "optimizer"):
            m.actor.optimizer.param_groups[0]['lr'] = finetune_lr
        if hasattr(m, "critic") and hasattr(m.critic, "optimizer"):
            m.critic.optimizer.param_groups[0]['lr'] = finetune_lr
        print(f"  📉  Learning rate verified at {finetune_lr} for Phase {phase}")

    # ── Load or create model ──────────────────────────────────────────
    # Scan for ALL checkpoints for this phase, try newest first
    import glob
    import re
    checkpoint_pattern = os.path.join(model_dir, f"{algo_name}_phase{phase}_*_steps.zip")
    
    def get_step_count(filepath):
        match = re.search(r'_(\d+)_steps\.zip$', filepath)
        return int(match.group(1)) if match else 0
        
    checkpoints = sorted(glob.glob(checkpoint_pattern), key=get_step_count, reverse=True)
    
    resumed = False
    
    # Priority 1: explicit --resume-from path
    if resume_from is not None:
        ckpt_name = resume_from.replace(".zip", "")
        print(f"\n  🎯  Explicit resume from: {ckpt_name}")
        try:
            model = AlgoClass.load(ckpt_name, env=env)
            test_obs = env.observation_space.sample()
            test_action, _ = model.predict(test_obs, deterministic=True)
            if not np.all(np.isfinite(test_action)):
                print(f"  ❌  Checkpoint corrupted (NaN actions), cannot resume!")
                env.close()
                sys.exit(1)
            _lower_lr(model)
            print(f"  ✅  Resumed from explicit checkpoint")
            resumed = True
        except Exception as e:
            print(f"  ❌  Failed to load checkpoint: {e}")
            env.close()
            sys.exit(1)
    
    # Priority 2: auto-discover checkpoints
    if not resumed:
        for ckpt_path in checkpoints:
            ckpt_name = ckpt_path.replace(".zip", "")
            print(f"\n  🔍  Found checkpoint: {os.path.basename(ckpt_path)}")
            try:
                model = AlgoClass.load(ckpt_name, env=env)
                test_obs = env.observation_space.sample()
                test_action, _ = model.predict(test_obs, deterministic=True)
                if not np.all(np.isfinite(test_action)):
                    print(f"  ❌  Checkpoint corrupted (NaN actions), skipping...")
                    continue
                _lower_lr(model)
                print(f"  ✅  Resumed from: {os.path.basename(ckpt_path)}")
                resumed = True
                break
            except Exception as e:
                print(f"  ❌  Failed to load checkpoint: {e}")
                continue
    
    if not resumed and phase == 1:
        print(f"\n  🆕  Creating fresh {algo_name.upper()} model (Phase 1)")
        model = create_fresh_model(algo_name, env, cfg)
    elif not resumed:
        prev_model_path = get_model_path(algo_name, phase - 1)
        full_prev_path = os.path.join(model_dir, f"{algo_name}_phase{phase - 1}")

        if os.path.exists(full_prev_path + ".zip"):
            print(f"\n  📂  Loading Phase {phase-1} model: {full_prev_path}")
            model = AlgoClass.load(full_prev_path, env=env)
            print(f"  🧠  Weights preserved from Phase {phase-1}")
            print(f"  🔄  Replay buffer reset for Phase {phase} data")
            _lower_lr(model)
        else:
            print(f"\n  ❌  Phase {phase-1} model not found at {full_prev_path}.zip")
            print(f"  Please train Phase {phase-1} first:")
            print(f"    python3 train_curriculum.py --algo {algo_name} "
                  f"--phase {phase-1}")
            env.close()
            sys.exit(1)

    print(f"\n  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = [
        NavigationMetricsCallback(
            eval_freq=2_000,
            log_prefix=f"{algo_name}_p{phase}",
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=5_000,
            save_path=model_dir,
            algo_name=f"{algo_name}_phase{phase}",
        ),
        SuccessRateStopCallback(
            success_threshold=0.90,  # Stop at 90% success rate
            min_episodes=50,
            check_freq=2_000,
            verbose=1,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────
    with TrainingTimer(f"{algo_name.upper()} Phase {phase}") as timer:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            tb_log_name=f"{algo_name}_phase{phase}",
            progress_bar=True,
            reset_num_timesteps=True,
        )

    # ── Save final model ──────────────────────────────────────────────
    final_path = os.path.join(model_dir, f"{algo_name}_phase{phase}")
    model.save(final_path)
    print(f"\n  💾  Phase {phase} model saved: {final_path}")

    # ── Save training metadata ────────────────────────────────────────
    save_training_results(f"{algo_name}_phase{phase}", {
        "algorithm": algo_name.upper(),
        "phase": phase,
        "phase_label": phase_info["label"],
        "total_timesteps": timesteps,
        "training_time_seconds": timer.elapsed,
        "model_path": final_path,
    })

    env.close()

    # ── Next phase instructions ───────────────────────────────────────
    if phase < 4:
        next_phase = phase + 1
        next_info = PHASE_DEFAULTS[next_phase]
        print(f"\n  📋  NEXT STEP — Phase {next_phase}: {next_info['label']}")
        print(f"  ┌─────────────────────────────────────────────────────┐")
        print(f"  │  1. Stop Webots (Ctrl+C on launch terminal)        │")
        print(f"  │  2. export CURRICULUM_PHASE={next_phase}                     │")
        print(f"  │  3. ros2 launch nav_env_pkg nav_launch.py          │")
        print(f"  │  4. python3 train_curriculum.py \\                  │")
        print(f"  │       --algo {algo_name} --phase {next_phase} "
              f"--timesteps {next_info['timesteps']}    │")
        print(f"  └─────────────────────────────────────────────────────┘")
    else:
        print(f"\n  🏆  ALL 4 PHASES COMPLETE for {algo_name.upper()}!")
        print(f"  Final model: {final_path}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum-based DRL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train TD3 through all 4 phases:
  python3 train_curriculum.py --algo td3 --phase 1 --timesteps 30000
  python3 train_curriculum.py --algo td3 --phase 2 --timesteps 40000
  python3 train_curriculum.py --algo td3 --phase 3 --timesteps 50000
  python3 train_curriculum.py --algo td3 --phase 4 --timesteps 60000

  # Train DQN Phase 1 only:
  python3 train_curriculum.py --algo dqn --phase 1 --timesteps 50000
        """)
    parser.add_argument("--algo", type=str, required=True,
                        choices=["dqn", "ddpg", "td3", "sac"],
                        help="Algorithm to train")
    parser.add_argument("--phase", type=int, required=True,
                        choices=[1, 2, 3, 4],
                        help="Curriculum phase (1-4)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override timesteps (default: phase-specific)")
    parser.add_argument("--resume-from", type=str, default=None,
                        dest="resume_from",
                        help="Explicit path to checkpoint to resume from "
                             "(without .zip extension)")

    args = parser.parse_args()

    timesteps = args.timesteps or PHASE_DEFAULTS[args.phase]["timesteps"].get(args.algo, 50_000)

    train_phase(args.algo, args.phase, timesteps, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
