"""
Base environment class with shared logic for feature extraction.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..config import (
    AVAILABLE_PASSES, ENCODING_OPS,
    MAX_PASS_STEPS, MAX_ENCODING_OPS, MAX_STEPS,
)
from ..pe_utils import calc_entropy, calc_printable_ratio
from ..compiler import apply_pass


class BaseEnv(gym.Env):
    """
    Base environment with shared observation space and feature extraction.
    """
    
    def _init_spaces(self):
        """Initialize action and observation spaces."""
        self.available_passes = AVAILABLE_PASSES.copy()
        self.encoding_ops = ENCODING_OPS.copy()
        self.max_steps = MAX_STEPS
        
        self.num_passes = len(self.available_passes)
        self.num_encoding_ops = len(self.encoding_ops)
        
        # Action space: [pass_selection, encoding_selection]
        self.action_space = spaces.MultiDiscrete([
            self.num_passes + 1,      # +1 for skip action
            self.num_encoding_ops + 1
        ])
        
        # Observation space setup
        self.pass_action_size = self.num_passes + 1
        self.enc_action_size = self.num_encoding_ops + 1
        self.pass_empty_idx = self.num_passes
        self.pass_skip_idx = self.num_passes
        self.enc_skip_idx = self.num_encoding_ops
        self.enc_empty_idx = self.num_encoding_ops

        obs_dim = (
            4 + 1 +  # Payload features + step remaining
            self.num_passes + 1 +  # Pass applied one-hot + pass remaining
            (MAX_PASS_STEPS * self.pass_action_size) +  # Pass sequence slots
            self.pass_action_size + 1 +  # Last pass action + applied flag
            self.num_encoding_ops + 1 +  # Encoding counts + enc remaining
            (MAX_ENCODING_OPS * self.enc_action_size) +  # Encoding sequence slots
            self.enc_action_size + 1 +  # Last enc action + added flag
            1  # Last step effect
        )
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        # Index lookup for speed
        self._pass_to_idx = {name: i for i, name in enumerate(self.available_passes)}
        self._enc_to_idx = {name: i for i, name in enumerate(self.encoding_ops)}
    
    def _init_state(self):
        """Initialize state variables."""
        self.current_bc: Optional[str] = None
        self.current_bits: int = 64
        self.current_payload_data: Optional[bytes] = None
        self.step_count: int = 0
        self.pass_history: List[str] = []
        self.encoding_sequence: List[str] = []
        
        # Observation state
        self.last_pass_action_idx = self.pass_skip_idx
        self.last_pass_applied = 0.0
        self.last_enc_action_idx = self.enc_skip_idx
        self.last_enc_added = 0.0
        self.last_step_effect = 0.0
    
    def _reset_state(self):
        """Reset state for new episode."""
        self.step_count = 0
        self.pass_history = []
        self.encoding_sequence = []
        
        self.last_pass_action_idx = self.pass_skip_idx
        self.last_pass_applied = 0.0
        self.last_enc_action_idx = self.enc_skip_idx
        self.last_enc_added = 0.0
        self.last_step_effect = 0.0
    
    def _process_action(self, action: np.ndarray) -> tuple:
        """
        Process action and update state.
        
        Returns:
            Tuple of (pass_applied, enc_added) flags.
        """
        pass_idx = action[0]
        enc_op_idx = action[1]

        pass_applied = 0.0
        enc_added = 0.0
        
        # Process pass action
        if pass_idx < self.num_passes:
            pass_name = self.available_passes[pass_idx]
            
            if pass_name not in self.pass_history and len(self.pass_history) < MAX_PASS_STEPS:
                new_bc = str(self.work_dir / f"{self._bc_prefix}_s{self.step_count}.bc")
                success = apply_pass(
                    self.current_bc, new_bc, pass_name,
                    self.delay_loop_plugin
                )
                
                if success:
                    self.current_bc = new_bc
                    self.pass_history.append(pass_name)
                    pass_applied = 1.0
        
        # Process encoding action
        if enc_op_idx < self.num_encoding_ops:
            enc_op = self.encoding_ops[enc_op_idx]
            if len(self.encoding_sequence) < MAX_ENCODING_OPS:
                self.encoding_sequence.append(enc_op)
                enc_added = 1.0

        # Update observation state
        self.last_pass_action_idx = int(pass_idx)
        self.last_pass_applied = float(pass_applied)
        self.last_enc_action_idx = int(enc_op_idx)
        self.last_enc_added = float(enc_added)
        self.last_step_effect = 1.0 if (pass_applied > 0.0 or enc_added > 0.0) else 0.0
        
        self.step_count += 1
        
        return pass_applied, enc_added
    
    def _extract_features(self) -> np.ndarray:
        """Extract observation features."""
        # Payload features
        payload_size = len(self.current_payload_data) if self.current_payload_data else 0
        payload_entropy = calc_entropy(self.current_payload_data) if self.current_payload_data else 0.0
        payload_bits = self.current_bits / 64.0
        printable_ratio = calc_printable_ratio(self.current_payload_data) if self.current_payload_data else 0.0
        
        payload_features = [
            min(payload_size / 1000000, 1.0),
            payload_entropy / 8.0,
            payload_bits,
            printable_ratio,
        ]

        # Progress/budget
        step_remaining = float(max(MAX_STEPS - self.step_count, 0) / MAX_STEPS)

        # Pass features
        pass_applied_oh = [0.0] * self.num_passes
        for p in self.pass_history:
            idx = self._pass_to_idx.get(p)
            if idx is not None:
                pass_applied_oh[idx] = 1.0

        pass_remaining = float(max(MAX_PASS_STEPS - len(self.pass_history), 0) / MAX_PASS_STEPS)

        # Pass sequence slots
        pass_seq = []
        for slot in range(MAX_PASS_STEPS):
            idx = self.pass_empty_idx
            if slot < len(self.pass_history):
                idx = self._pass_to_idx.get(self.pass_history[slot], self.pass_empty_idx)
            onehot = [0.0] * self.pass_action_size
            if 0 <= idx < self.pass_action_size:
                onehot[idx] = 1.0
            else:
                onehot[self.pass_empty_idx] = 1.0
            pass_seq.extend(onehot)

        # Last pass action
        last_pass_onehot = [0.0] * self.pass_action_size
        last_pass_idx = int(self.last_pass_action_idx)
        if 0 <= last_pass_idx < self.pass_action_size:
            last_pass_onehot[last_pass_idx] = 1.0
        else:
            last_pass_onehot[self.pass_skip_idx] = 1.0

        # Encoding features
        enc_counts = [0.0] * self.num_encoding_ops
        for op in self.encoding_sequence:
            idx = self._enc_to_idx.get(op)
            if idx is not None:
                enc_counts[idx] += 1.0
        if MAX_ENCODING_OPS > 0:
            enc_counts = [min(c / MAX_ENCODING_OPS, 1.0) for c in enc_counts]

        enc_remaining = float(max(MAX_ENCODING_OPS - len(self.encoding_sequence), 0) / MAX_ENCODING_OPS)

        # Encoding sequence slots
        enc_seq = []
        for slot in range(MAX_ENCODING_OPS):
            idx = self.enc_empty_idx
            if slot < len(self.encoding_sequence):
                idx = self._enc_to_idx.get(self.encoding_sequence[slot], self.enc_empty_idx)
            onehot = [0.0] * self.enc_action_size
            if 0 <= idx < self.enc_action_size:
                onehot[idx] = 1.0
            else:
                onehot[self.enc_empty_idx] = 1.0
            enc_seq.extend(onehot)

        # Last encoding action
        last_enc_onehot = [0.0] * self.enc_action_size
        last_enc_idx = int(self.last_enc_action_idx)
        if 0 <= last_enc_idx < self.enc_action_size:
            last_enc_onehot[last_enc_idx] = 1.0
        else:
            last_enc_onehot[self.enc_skip_idx] = 1.0

        # Combine all features
        all_features = (
            payload_features +
            [step_remaining] +
            pass_applied_oh + [pass_remaining] + pass_seq + last_pass_onehot + [float(self.last_pass_applied)] +
            enc_counts + [enc_remaining] + enc_seq + last_enc_onehot + [float(self.last_enc_added)] +
            [float(self.last_step_effect)]
        )
        
        return np.array(all_features, dtype=np.float32)
