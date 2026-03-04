"""
Configuration constants for the evasion framework.
"""

from typing import Dict, List

# ======================== Pass Configuration ========================
AVAILABLE_PASSES: List[str] = [
    'sroa',           # 0: Scalar Replacement of Aggregates
    'gvn',            # 1: Global Value Numbering
    'simplifycfg',    # 2: Simplify Control Flow Graph
    'loop-rotate',    # 3: Loop Rotation
    'licm',           # 4: Loop Invariant Code Motion
    'globalopt',      # 5: Global Optimization
    'delay-loop',     # 6: Custom Delay Loop Pass
]

# ======================== Encoding Configuration ========================
ENCODING_OPS: List[str] = [
    'xor_full',   # 0: 100% XOR
    'xor_75',     # 1: 75% XOR
    'xor_50',     # 2: 50% XOR
    'xor_25',     # 3: 25% XOR
    'pad_front',  # 4: Front Padding
    'pad_back',   # 5: Back Padding
    'shuffle',    # 6: Byte Shuffle
]

# ======================== RL Environment Parameters ========================
MAX_PASS_STEPS: int = 3       # Maximum number of passes per episode
MAX_ENCODING_OPS: int = 5     # Maximum number of encoding operations per episode
MAX_STEPS: int = 5            # Maximum steps per episode

# ======================== Encoding Constants ========================
XOR_KEY: int = 0x42           # XOR key used for encoding

# ======================== Compiler Configuration ========================
# Relative paths from project root
INCLUDE_DIR_32: str = "toolchain/mingw32/include"
INCLUDE_DIR_64: str = "toolchain/mingw64/include"
LIB_DIR_32: str = "toolchain/mingw32/lib"
LIB_DIR_64: str = "toolchain/mingw64/lib"

# Compiler targets
TARGET_32: str = "i686-w64-mingw32"
TARGET_64: str = "x86_64-w64-mingw32"

# ======================== AV Scanner Configuration ========================
# Replace these with the actual addresses of your AV scanning services.
# Each service should expose a REST endpoint that accepts multipart/form-data
# file uploads and returns a JSON response with a "result" field
# (0 = clean, 1 = detected).
ENGINE_ADDRESS: Dict[str, str] = {
    'Avast':      'http://<AVAST_HOST>:5000/upload_sync',
    'ClamAV':     'http://<CLAMAV_HOST>:5000/upload_sync',
    'ESET':       'http://<ESET_HOST>:5000/upload_sync',
    'DrWeb':      'http://<DRWEB_HOST>:5001/upload_sync',
    'Kaspersky':  'http://<KASPERSKY_HOST>:5000/upload_sync',
    'HuoRong':    'http://<HUORONG_HOST>:5001/upload_sync',
    'MicroSoft':  'http://<MICROSOFT_HOST>:5000/upload_sync',
}

MODEL_ADDRESS: Dict[str, str] = {
    'invincea':   'http://<MODEL_HOST>:5030/upload_sync',
    'malconv':    'http://<MODEL_HOST>:7004/upload_sync',
    'avastnet':   'http://<MODEL_HOST>:5000/upload_sync',
    'ember_gbdt': 'http://<MODEL_HOST>:5010/upload_sync',
    'ember_svm':  'http://<MODEL_HOST>:5020/upload_sync',
    'malconv2':   'http://<MODEL_HOST>:5005/upload_sync',
    'markovnet':  'http://<MODEL_HOST>:5006/upload_sync',
}

# ======================== Training Parameters ========================
# buffer_size is the total rollout buffer size across ALL environments.
# n_steps per env = buffer_size // num_envs, so training is invariant to num_envs.
DEFAULT_TRAINING_CONFIG = {
    'learning_rate': 3e-4,
    'buffer_size': 2048,   # total steps collected per update (all envs combined)
    'batch_size': 256,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.03,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'total_timesteps': 100000,
}

# ======================== Reward Configuration ========================
REWARD_BYPASS: float = 10.0           # Base reward for successful bypass
REWARD_STEP_BONUS: float = 0.0        # Bonus per remaining step when bypassing
REWARD_FAIL_EPISODE: float = 0.0     # Penalty for failing the episode
REWARD_STEP_PENALTY: float = 0.0     # Small penalty per step
