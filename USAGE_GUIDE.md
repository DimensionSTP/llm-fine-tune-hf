# Usage Guide

## Purpose

Practical runbook for training and inference flows in `llm-fine-tune-hf`.

## Install

```bash
python -m pip install uv==0.10.12
uv pip install "setuptools>=68" wheel
uv pip install \
    --override requirements-overrides.txt \
    --torch-backend=cu129 \
    --no-build-isolation \
    -r requirements.txt
```

or

```bash
python -m pip install uv==0.10.12
uv pip install "setuptools>=68" wheel
uv pip install \
    --override requirements-overrides.txt \
    --torch-backend=cu129 \
    --no-build-isolation \
    .
uv pip install \
    --override requirements-overrides.txt \
    --torch-backend=cu129 \
    --no-build-isolation \
    -e .
```

Optional GPU dependency (`flash-attn`):

```bash
uv pip install \
    --override requirements-overrides.txt \
    --torch-backend=cu129 \
    --no-build-isolation \
    ".[gpu]"
# or
python -m pip install --no-build-isolation "flash-attn @ git+https://github.com/Dao-AILab/flash-attention.git@060c9188beec3a8b62b33a3bfa6d5d2d44975fab"
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

Training automatically allocates `run_id` values such as `run-0001` under the method/model/data checkpoint path. `run_manifest.json` and the full `resolved_config.yaml` are written before preflight, tracking, and setup; `training_args.json` is added after argument instantiation. The manifest records `prepared`, `running`, `completed`, `failed`, or `interrupted` with the last `preflight`, `setup`, `training`, `saving`, or `completed` stage. Config values remain in `resolved_config.yaml`, effective trainer arguments remain in `training_args.json`, and the manifest keeps observed runtime facts and relative artifact references without duplicating either file. Runtime batch-size fields stay out of the checkpoint path. For distributed or multi-node runs, the manifest records planned and observed distributed, device, batch, and DataLoader runtime.

Single-turn test modes write the canonical `${connected_dir}/tests/${model_detail}/${dataset_name}.json` result and `${dataset_name}_manifest.json` companion. Multi-turn vLLM testing writes `${dataset_name}_multi_turn.jsonl` and `${dataset_name}_multi_turn_manifest.json` in the same model-detail directory. Companion manifests store the full resolved config once and add only the resolved test files, active data encoder, runtime backend, effective device map or tensor-parallel size, and actual generation parameters.

MLflow is the default tracking backend. Use `tracking=wandb` for W&B or `tracking=mlflow_server` for the remote endpoint in `MLFLOW_TRACKING_URI`. The server profile leaves artifact location selection to the server and uses `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` when the endpoint requires MLflow basic authentication. Train runs write the selected backend and generated tracking UUID to `${output_dir}/tracking_metadata.json`; its location under the artifact run provides the mapping without copying config fields. Normal completion, ordinary exceptions, and `KeyboardInterrupt` or `SystemExit` end MLflow runs as `FINISHED`, `FAILED`, and `KILLED`, respectively.

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

- Data/tokenization: `is_sft`, `is_preprocessed`, `left_padding`, `max_length`, `response_end_template`, `truncation_mode`, `pad_to_multiple_of`
- Dataset paths: `data_path`, `dataset_subdir`, `dataset_file_path`, `test_dataset_subdir`, `test_dataset_file_path`
- SFT label mask validation: `sft_label_mask.validation_enabled`, `sft_label_mask.validation_mode`
- SFT loss: `sft_loss_type=nll`; set `training_arguments.use_liger_kernel=false` before selecting `chunked_nll`
- SFT padding: `sft_padding_strategy=dynamic`; use `max_length` for fixed padding
- SFT Liger kernel: `training_arguments.use_liger_kernel=true`
- Training strategy: `strategy=deepspeed`
- Distributed launch: `distributed.enabled`, `distributed.num_machines`, `distributed.num_processes_per_machine`, `distributed.machine_rank`, `distributed.main_process_ip`, `distributed.main_process_port`
- Tracking backend: `tracking=wandb`, `tracking=mlflow`, `tracking=mlflow_server`
- PEFT/quantization: `is_quantized`, `is_peft`
- LoRA merge: `merge_max_shard_size`, `merge_pack_qwen_moe_experts`
- GRPO/vLLM: `use_vllm`, `vllm_mode`, `vllm_sync_strategy`
- VLM image augmentation: `image_augmentation.enabled`, `image_augmentation.probability`
- VLM image paths: `dataset_image.image_root_dir`, `dataset_image.convert_unsupported_extensions`; base64 images are decoded to PIL images when needed
- Reward extraction: `reward.extraction_profile`
- Retrieval rewards: `reward_embedding.preserved_env_keys`, `reward_embedding.isolated_env_keys`
- Model card/upload metadata: `upload_user`, `model_type`

## Validation Checklist

1. Mode command resolves Hydra config without errors.
2. Output artifacts are created for selected mode.
3. If using `flash-attn`, installation succeeds in current CUDA/toolchain environment.
4. Changes are reflected in changelog and release notes before release.
