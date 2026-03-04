# MorphRL

**MorphRL** is a reinforcement learning framework for generating adversarial PE (Portable Executable) files that evade malware detection. It trains a PPO agent to intelligently compose LLVM compiler optimization passes and binary encoding transformations to morph malware loaders while preserving functionality.

## Overview

MorphRL operates as a sequential decision-making problem:

1. A malware sample (payload) is provided as input.
2. The RL agent selects a sequence of **LLVM IR passes** and **binary encoding operations** to apply to a loader stub.
3. The transformed loader is compiled and the payload is attached.
4. The resulting PE file is submitted to an AV scanner for evaluation.
5. The agent receives a reward signal based on evasion success.

### Action Space

**LLVM Passes:**
| ID | Pass | Description |
|----|------|-------------|
| 0 | `sroa` | Scalar Replacement of Aggregates |
| 1 | `gvn` | Global Value Numbering |
| 2 | `simplifycfg` | Simplify Control Flow Graph |
| 3 | `loop-rotate` | Loop Rotation |
| 4 | `licm` | Loop Invariant Code Motion |
| 5 | `globalopt` | Global Optimization |
| 6 | `delay-loop` | Custom Delay Loop Injection |

**Encoding Operations:**
| ID | Operation | Description |
|----|-----------|-------------|
| 0 | `xor_full` | 100% XOR encoding |
| 1 | `xor_75` | 75% XOR encoding |
| 2 | `xor_50` | 50% XOR encoding |
| 3 | `xor_25` | 25% XOR encoding |
| 4 | `pad_front` | Front padding |
| 5 | `pad_back` | Back padding |
| 6 | `shuffle` | Byte shuffle |

## Requirements

### System Dependencies

- Linux (tested on Debian/Ubuntu)
- Clang/LLVM 20
- MinGW-w64 (for cross-compiling PE files targeting Windows)

### Python Environment

Using conda (recommended):

```bash
conda env create -f environment.yml
conda activate rlevade
```

Or manually:

```bash
pip install stable_baselines3==2.7.1 numpy requests rich tqdm
```

### MinGW Toolchain

The `toolchain/` directory must contain `mingw32/` and `mingw64/` subdirectories with MinGW-w64 cross-compilation toolchains. You can install them via:

```bash
apt install gcc-mingw-w64 g++-mingw-w64
```

Then update `INCLUDE_DIR_32/64` and `LIB_DIR_32/64` paths in `src/config.py` accordingly.

### AV Scanning Services

MorphRL requires REST-based AV scanning endpoints. Configure the addresses in `src/config.py`:

```python
ENGINE_ADDRESS = {
    'ClamAV': 'http://<YOUR_HOST>:5000/upload_sync',
    ...
}
MODEL_ADDRESS = {
    'malconv': 'http://<YOUR_HOST>:7004/upload_sync',
    ...
}
```

Each endpoint must accept `multipart/form-data` file uploads and return:

```json
{ "result": 0 }   // 0 = clean, 1 = detected
```

## Project Structure

```
MorphRL/
├── train.py              # Training script
├── test.py               # Evaluation script
├── environment.yml       # Conda environment
├── src/
│   ├── config.py         # All hyperparameters and addresses
│   ├── scanner.py        # AV scanner client
│   ├── compiler.py       # LLVM IR compilation pipeline
│   ├── encoding.py       # Binary encoding operations
│   ├── pe_utils.py       # PE file utilities
│   ├── callbacks.py      # SB3 training callbacks
│   └── envs/
│       ├── base.py       # Base RL environment
│       ├── train_env.py  # Training environment
│       └── test_env.py   # Testing environment
├── stub/
│   └── src/              # Loader stub C source code
└── llvm/
    └── passes/           # Custom LLVM pass plugins (DelayLoop.so)
```

## Usage

### Training

```bash
# Train against an ML model detector
python train.py --av-type model --av-name malconv --num-envs 16

# Train against a traditional AV engine
python train.py --av-type engine --av-name ClamAV --num-envs 8

# Custom dataset and timesteps
python train.py \
    --av-type model \
    --av-name malconv \
    --sample-dir data/train_samples \
    --num-envs 16 \
    --total-timesteps 200000
```

The trained policy is saved to `policies/<av_name>_ppo.zip`.

**Note:** `n_steps` per environment is automatically computed as `buffer_size // num_envs` (default `buffer_size=2048`), so training dynamics remain consistent regardless of the number of parallel environments.

### Evaluation

```bash
# Evaluate a trained policy
python test.py --av-type model --av-name malconv --max-samples 500

# Specify a custom model path and test set
python test.py \
    --av-type model \
    --av-name malconv \
    --model-path policies/malconv_ppo \
    --sample-dir data/test_samples \
    --max-samples 1000
```

Adversarial samples are saved to `outputs/test/<av_name>/adversarial/`.

### Data Layout

```
data/
├── train_samples/    # PE malware samples for training
└── test_samples/     # PE malware samples for evaluation
```

## License

This project is released for academic research purposes only. Do not use it for malicious purposes.
