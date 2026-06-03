# RAW-Mal

**RAW-Mal** is a black-box adversarial evasion framework for Windows malware. It wraps malware in a custom PE loader and applies two complementary transformation channels: semantics-preserving LLVM passes that restructure the loader binary, and reversible byte-level encodings that reshape the embedded malware. A PPO agent learns to select transformations via hard-label detector feedback.


## Requirements

### System Dependencies

- Linux (tested on Debian/Ubuntu)
- Clang/LLVM 20
- MinGW-w64 (for cross-compiling PE files targeting Windows)

### Python Environment

```bash
conda env create -f environment.yml
conda activate rawmal
```

### MinGW Toolchain

Extract the pre-packaged toolchain archive:

```bash
tar -xzf toolchain.tar.gz
```

### AV Scanning Services

RAW-Mal requires REST-based AV scanning endpoints. Configure the addresses in `src/config.py`:

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
{ "result": 0 }   // 0 = undetected, 1 = detected
```

## Project Structure

```
RAW-Mal/
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

# Train against a commercial AV engine
python train.py --av-type engine --av-name ClamAV --num-envs 16

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
python test.py --av-type model --av-name malconv --max-samples 2000

# Specify a custom model path and test set
python test.py \
    --av-type model \
    --av-name malconv \
    --model-path policies/malconv_ppo \
    --sample-dir data/test_samples \
    --max-samples 2000
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
