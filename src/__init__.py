"""
RLEvade - Reinforcement Learning based Malware Evasion Framework

This package provides tools for training and testing RL-based evasion strategies
against antivirus models and engines.
"""

from .config import (
    AVAILABLE_PASSES,
    ENCODING_OPS,
    MAX_PASS_STEPS,
    MAX_ENCODING_OPS,
    MAX_STEPS,
    XOR_KEY,
)
from .pe_utils import (
    detect_pe_bits,
    get_pe_image_base,
    get_pe_subsystem,
)
from .encoding import encode_payload_sequence
from .compiler import compile_stub_to_ir, compile_to_exe
from .scanner import build_scanner, LocalOnlineAntivirus

__version__ = "1.0.0"
__all__ = [
    # Config
    "AVAILABLE_PASSES",
    "ENCODING_OPS",
    "MAX_PASS_STEPS",
    "MAX_ENCODING_OPS",
    "MAX_STEPS",
    "XOR_KEY",
    # PE Utils
    "detect_pe_bits",
    "get_pe_image_base",
    "get_pe_subsystem",
    # Encoding
    "encode_payload_sequence",
    # Compiler
    "compile_stub_to_ir",
    "compile_to_exe",
    # Scanner
    "build_scanner",
    "LocalOnlineAntivirus",
]

# Lazy imports for RL components (require stable_baselines3)
def __getattr__(name):
    if name == "SuccessRateCallback":
        from .callbacks import SuccessRateCallback
        return SuccessRateCallback
    elif name == "TrainEnv":
        from .envs import TrainEnv
        return TrainEnv
    elif name == "TestEnv":
        from .envs import TestEnv
        return TestEnv
    elif name == "make_env":
        from .envs import make_env
        return make_env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
