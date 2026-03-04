#!/usr/bin/env python3
"""
Training script for the PE loader evasion RL agent.

Usage:
    python train.py --av-type model --av-name avastnet --num-envs 16
"""

import os
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from src.envs import make_env
from src.callbacks import SuccessRateCallback
from src.compiler import compile_stub_to_ir
from src.config import DEFAULT_TRAINING_CONFIG
from src.scanner import build_scanner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL agent for AV evasion"
    )
    parser.add_argument(
        "--av-type", type=str, default="model",
        choices=["model", "engine"],
        help="Target AV type (default: model)"
    )
    parser.add_argument(
        "--av-name", type=str, default="avastnet",
        help="Target AV name (default: avastnet)"
    )
    parser.add_argument(
        "--sample-dir", type=str,
        default="data/train_samples",
        help="Directory containing training samples (default: data/train_samples)"
    )
    parser.add_argument(
        "--num-envs", type=int, default=16,
        help="Number of parallel environments (default: 16)"
    )
    parser.add_argument(
        "--total-timesteps", type=int, 
        default=DEFAULT_TRAINING_CONFIG['total_timesteps'],
        help="Total training timesteps (default: 400000)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Paths
    stub_dir = "stub"
    source_bc_32 = f"{stub_dir}/build/stub_32.bc"
    source_bc_64 = f"{stub_dir}/build/stub_64.bc"
    delay_loop_plugin = str(project_dir / "llvm" / "passes" / "DelayLoop.so")
    
    # Check and compile IR files
    if not os.path.exists(source_bc_32):
        print(f"\nCompiling 32-bit IR...")
        if not compile_stub_to_ir(32, stub_dir):
            return
    
    if not os.path.exists(source_bc_64):
        print(f"\nCompiling 64-bit IR...")
        if not compile_stub_to_ir(64, stub_dir):
            return
    
    if not os.path.exists(args.sample_dir):
        print(f"Error: Sample directory not found: {args.sample_dir}")
        return
    
    buffer_size = DEFAULT_TRAINING_CONFIG['buffer_size']
    n_steps = max(1, buffer_size // args.num_envs)

    print(f"\nConfiguration:")
    print(f"  32-bit IR: {source_bc_32}")
    print(f"  64-bit IR: {source_bc_64}")
    print(f"  Training samples: {args.sample_dir}")
    print(f"  Target AV: {args.av_type}:{args.av_name}")
    print(f"  Parallel environments: {args.num_envs}")
    print(f"  Buffer size (total): {buffer_size}  (n_steps per env: {n_steps})")
    print(f"  Total timesteps: {args.total_timesteps}")
    
    # Check AV online
    print(f"\n[*] Checking AV service: {args.av_type}:{args.av_name}")
    scanner = build_scanner(args.av_type, args.av_name)
    scanner.check_online()
    print(f"  AV service online")
    
    # Create environments
    print("\nCreating environments...")
    
    if args.num_envs > 1:
        env = SubprocVecEnv([
            make_env(
                source_bc_32, source_bc_64, args.sample_dir,
                args.av_type, args.av_name, i,
                delay_loop_plugin=delay_loop_plugin
            )
            for i in range(args.num_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(
                source_bc_32, source_bc_64, args.sample_dir,
                args.av_type, args.av_name, 0,
                delay_loop_plugin=delay_loop_plugin
            )
        ])
    
    print(f"  {args.num_envs} environments ready")
    
    # Create model
    print("\nCreating PPO Agent...")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=DEFAULT_TRAINING_CONFIG['learning_rate'],
        n_steps=n_steps,
        batch_size=DEFAULT_TRAINING_CONFIG['batch_size'],
        n_epochs=DEFAULT_TRAINING_CONFIG['n_epochs'],
        gamma=DEFAULT_TRAINING_CONFIG['gamma'],
        gae_lambda=DEFAULT_TRAINING_CONFIG['gae_lambda'],
        clip_range=DEFAULT_TRAINING_CONFIG['clip_range'],
        ent_coef=DEFAULT_TRAINING_CONFIG['ent_coef'],
        vf_coef=DEFAULT_TRAINING_CONFIG['vf_coef'],
        max_grad_norm=DEFAULT_TRAINING_CONFIG['max_grad_norm'],
        verbose=1,
    )
    
    # Callback
    callback = SuccessRateCallback()
    
    # Training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        
        # Statistics
        stats = callback.get_final_stats()
        if stats['total_episodes'] > 0:
            print(f"\nFinal Statistics:")
            print(f"  Total Episodes: {stats['total_episodes']}")
            print(f"  Successful Bypasses: {stats['successes']}")
            print(f"  Success Rate: {stats['success_rate']:.2f}%")
        
        # Save policy
        policies_dir = Path("policies")
        policies_dir.mkdir(exist_ok=True)
        policy_name = f"{args.av_name}_ppo"
        policy_path = policies_dir / policy_name
        model.save(str(policy_path))
        print(f"\nPolicy saved: {policy_path}.zip")
        
        # Plot training curve
        curve_path = policies_dir / f"{args.av_name}_training_curve.png"
        callback.plot_training_curve(str(curve_path), args.av_name)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        # Save curve even if interrupted
        if callback.total_episodes > 0:
            policies_dir = Path("policies")
            policies_dir.mkdir(exist_ok=True)
            curve_path = policies_dir / f"{args.av_name}_training_curve_interrupted.png"
            callback.plot_training_curve(str(curve_path), args.av_name)
    finally:
        env.close()


if __name__ == "__main__":
    main()
