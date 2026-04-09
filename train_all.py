#!/usr/bin/env python3
"""
train_all.py — Run all 4 algorithms sequentially
==================================================
Usage:
    python3 train_all.py                        # Train all with defaults
    python3 train_all.py --algo td3 sac         # Train specific ones
    python3 train_all.py --timesteps 10000      # Quick test run
"""
import argparse, sys, time

def main():
    parser = argparse.ArgumentParser(description="Train all DRL algorithms")
    parser.add_argument("--algo", nargs="+", default=["dqn", "ddpg", "td3", "sac"])
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total_start = time.time()

    print("=" * 60)
    print("  🏁  DRL Navigation — Multi-Algorithm Training")
    print(f"  Algorithms: {', '.join(a.upper() for a in args.algo)}")
    if args.timesteps:
        print(f"  Timesteps override: {args.timesteps:,}")
    print("=" * 60)

    results = {}

    if "dqn" in args.algo:
        from train_dqn import train_dqn
        from config import DQNConfig
        cfg = DQNConfig(seed=args.seed)
        if args.timesteps: cfg.total_timesteps = args.timesteps
        train_dqn(cfg)
        results["dqn"] = "✅"

    if "ddpg" in args.algo:
        from train_ddpg import train_ddpg
        from config import DDPGConfig
        cfg = DDPGConfig(seed=args.seed)
        if args.timesteps: cfg.total_timesteps = args.timesteps
        train_ddpg(cfg)
        results["ddpg"] = "✅"

    if "td3" in args.algo:
        from train_td3 import train_td3
        from config import TD3Config
        cfg = TD3Config(seed=args.seed)
        if args.timesteps: cfg.total_timesteps = args.timesteps
        train_td3(cfg)
        results["td3"] = "✅"

    if "sac" in args.algo:
        from train_sac import train_sac
        from config import SACConfig
        cfg = SACConfig(seed=args.seed)
        if args.timesteps: cfg.total_timesteps = args.timesteps
        train_sac(cfg)
        results["sac"] = "✅"

    elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("  🏁  ALL TRAINING COMPLETE")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    for algo, status in results.items():
        print(f"    {algo.upper()}: {status}")
    print("=" * 60)
    print("\n  Next: python3 evaluate.py")
    print("  TensorBoard: tensorboard --logdir logs/")

if __name__ == "__main__":
    main()
