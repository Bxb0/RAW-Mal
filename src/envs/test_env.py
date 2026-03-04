"""
Test environment for evaluating trained RL policies.
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np

from .base import BaseEnv


class TestEnv(BaseEnv):
    """
    Test environment for evaluating trained RL policies.
    
    This environment mirrors TrainEnv but is designed for inference
    rather than training. It doesn't require samples or scanner initialization.
    """
    
    def __init__(
        self,
        source_bc_32: str,
        source_bc_64: str,
        av_name: str = "test",
        delay_loop_plugin: Optional[str] = None
    ):
        """
        Initialize the test environment.
        
        Args:
            source_bc_32: Path to 32-bit stub bitcode.
            source_bc_64: Path to 64-bit stub bitcode.
            av_name: Name of the AV target (for working directory).
            delay_loop_plugin: Path to DelayLoop.so plugin.
        """
        super().__init__()
        
        self.source_bc_32 = source_bc_32
        self.source_bc_64 = source_bc_64
        self.delay_loop_plugin = delay_loop_plugin
        self._bc_prefix = "test"  # Prefix for bc files
        
        # Initialize spaces and state
        self._init_spaces()
        self._init_state()
        
        # Work directory
        self.base_work_dir = Path("outputs") / "test" / av_name
        self.base_work_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = self.base_work_dir / "work"
        self.work_dir.mkdir(exist_ok=True)
        
    def set_sample(self, payload_data: bytes, bits: int) -> None:
        """Set the current test sample."""
        self.current_payload_data = payload_data
        self.current_bits = bits
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for a new episode."""
        super().reset(seed=seed)
        
        source_bc = self.source_bc_32 if self.current_bits == 32 else self.source_bc_64
        self.current_bc = str(self.work_dir / f"test_{self.current_bits}bit.bc")
        subprocess.run(["cp", source_bc, self.current_bc], 
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
        
        return self._extract_features(), 0.0, False, truncated, {}
