# Usage Guide

## Purpose

Practical runbook for training and inference flows in `llm-fine-tune-hf`.

## Install

```bash
pip install --no-build-isolation -r requirements.txt
```

or

```bash
pip install --no-build-isolation .
pip install --no-build-isolation -e .
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

Training automatically allocates `run_id` values such as `run-0001` under the method/model/data checkpoint path and writes `run_manifest.json`, `resolved_config.yaml`, and `training_args.json` under `output_dir` before model construction. Runtime batch-size fields stay in metadata instead of the checkpoint path.

## Script-based Execution

- `bash scripts/preprocessing/preprocess.sh`
- `bash scripts/preprocessing/preprocess_dataset.sh`
- `bash scripts/train/train.sh`
- `bash scripts/train/async_grpo_train.sh`
- `bash scripts/postprocessing/merge_lora.sh`
- `bash scripts/test/test.sh`
- `bash scripts/test/test_large.sh`
- `bash scripts/test/test_vllm.sh`
- `bash scripts/test/test_vllm_multi_turn.sh`

Async GRPO server flow:

```bash
# script-managed server start/stop
# requirement: even number of GPUs and >=2 GPUs
bash scripts/train/async_grpo_train.sh
```

```bash
# config-only path (without train script)
# world_size=1: half GPUs for vLLM and half GPUs for trainer
# world_size=2: rank0=trainer, rank1=vLLM server
python main.py --config-name=async_grpo.yaml mode=train
```

```bash
# fallback cleanup for script-managed server
if [ -f /tmp/async_grpo_vllm_server.pid ]; then kill "$(cat /tmp/async_grpo_vllm_server.pid)"; fi
```

Postprocessing artifact paths:

```bash
bash scripts/postprocessing/merge_lora.sh
bash scripts/postprocessing/upload_to_hf_hub.sh
bash scripts/postprocessing/upload_all_to_hf_hub.sh
```

Postprocessing scripts keep `run_id` as a script-local variable. The Python entrypoint resolves the artifact path from config-composed `output_base_dir` and `run_id`.

## Common Runtime Options

- Data/tokenization: `is_sft`, `is_preprocessed`, `left_padding`, `max_length`, `response_end_template`
- SFT loss: `sft_loss_type`
- Training strategy: `strategy=deepspeed`
- PEFT/quantization: `is_quantized`, `is_peft`
- LoRA merge: `merge_max_shard_size`, `merge_pack_qwen_moe_experts`
- GRPO/vLLM: `use_vllm`, `vllm_mode`, `vllm_sync_strategy`
- VLM image augmentation: `image_augmentation.enabled`, `image_augmentation.probability`
- Reward extraction: `reward.extraction_profile`
- Retrieval rewards: `reward_embedding.preserved_env_keys`, `reward_embedding.isolated_env_keys`
- Model card/upload metadata: `upload_user`, `model_type`

## Validation Checklist

1. Mode command resolves Hydra config without errors.
2. Output artifacts are created for selected mode.
3. If using `flash-attn`, installation succeeds in current CUDA/toolchain environment.
4. Changes are reflected in changelog and release notes before release.
