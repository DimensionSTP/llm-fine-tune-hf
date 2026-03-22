# Usage Guide

## Purpose

Practical runbook for training and inference flows in `llm-fine-tune-hf`.

## Install

```bash
pip install -r requirements.txt
```

or

```bash
pip install .
pip install -e .
```

Optional GPU dependency (`flash-attn`):

```bash
pip install ".[gpu]"
# or
python -m pip install "flash-attn @ git+https://github.com/Dao-AILab/flash-attention.git@060c9188beec3a8b62b33a3bfa6d5d2d44975fab"
```

## Required Environment Variables

- `PROJECT_DIR`
- `CONNECTED_DIR`
- `DEVICES`
- `HF_HOME`
- `USER_NAME`

## Supported Modes (`main.py`)

- `train`
- `test`
- `test_large`
- `test_vllm`
- `test_vllm_multi_turn`

Run examples:

```bash
python main.py mode=train
python main.py mode=test
python main.py mode=test_large
python main.py mode=test_vllm
python main.py mode=test_vllm_multi_turn
```

## Script-based Execution

- `bash scripts/preprocessing/preprocess.sh`
- `bash scripts/preprocessing/preprocess_dataset.sh`
- `bash scripts/train/train.sh`
- `bash scripts/test/test.sh`
- `bash scripts/test/test_large.sh`
- `bash scripts/test/test_vllm.sh`
- `bash scripts/test/test_vllm_multi_turn.sh`

## Common Runtime Options

- Data/tokenization: `is_sft`, `is_preprocessed`, `left_padding`, `max_length`
- Training strategy: `strategy=deepspeed`
- PEFT/quantization: `is_quantized`, `is_peft`
- Model card/upload metadata: `upload_user`, `model_type`

## Validation Checklist

1. Mode command resolves Hydra config without errors.
2. Output artifacts are created for selected mode.
3. If using `flash-attn`, installation succeeds in current CUDA/toolchain environment.
4. Changes are reflected in changelog and release notes before release.
