# Usage Guide

## Purpose

Practical runbook for training and inference flows in `llm-fine-tune-hf`.

## Install

```bash
conda install -c nvidia "cuda-nvcc=12.9" -y
conda install -c defaults "libcurand=10.3.10.19" "libcurand-dev=10.3.10.19" -y
python -m pip install uv==0.10.12
uv pip install --torch-backend=cu129 "torch==2.13.0" "setuptools>=68" wheel packaging psutil ninja
CUDA_HOME="${CONDA_PREFIX}" \
CUDACXX="${CONDA_PREFIX}/bin/nvcc" \
CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include${CPATH:+:${CPATH}}" \
uv pip install \
    --override requirements-overrides.txt \
    --torch-backend=cu129 \
    --extra-index-url https://flashinfer.ai/whl/cu129 \
    --no-build-isolation \
    -r requirements.txt
```

The CUDA 12.9 compiler and pinned FlashInfer CUDA 12.9 JIT cache keep Blackwell sampler initialization from falling back to an older system CUDA toolkit. The pinned cuRAND runtime and development package provide the CUDA libraries and headers required to build DeepSpeed CPUAdam.

or

```bash
conda install -c nvidia "cuda-nvcc=12.9" -y
conda install -c defaults "libcurand=10.3.10.19" "libcurand-dev=10.3.10.19" -y
python -m pip install uv==0.10.12
uv pip install --torch-backend=cu129 "torch==2.13.0" "setuptools>=68" wheel packaging psutil ninja
CUDA_HOME="${CONDA_PREFIX}" \
CUDACXX="${CONDA_PREFIX}/bin/nvcc" \
CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include${CPATH:+:${CPATH}}" \
uv pip install \
    --override requirements-overrides.txt \
    --torch-backend=cu129 \
    --extra-index-url https://flashinfer.ai/whl/cu129 \
    --no-build-isolation \
    .
CUDA_HOME="${CONDA_PREFIX}" \
CUDACXX="${CONDA_PREFIX}/bin/nvcc" \
CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include${CPATH:+:${CPATH}}" \
uv pip install \
    --override requirements-overrides.txt \
    --torch-backend=cu129 \
    --extra-index-url https://flashinfer.ai/whl/cu129 \
    --no-build-isolation \
    -e .
```

Both installation paths include FlashAttention 2 and causal-conv1d from the same `requirements.txt` declarations; `pyproject.toml` reads that file dynamically. Install Torch and the build tools first, then install the full requirements or project with the active Conda CUDA compiler and headers. This build-time ordering is separate from runtime compatibility and does not require an optional GPU extra or separate FA2 installation. See [CUDA extension builds](README.md#cuda-extension-builds) for compiler requirements; rebuild both extensions after changing Torch.

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
- `test_decision`
- `test_vllm`
- `test_vllm_multi_turn`

Run examples:

```bash
python main.py mode=train
python main.py mode=test
python main.py mode=test_large
python main.py --config-name=decision.yaml mode=test_decision is_peft=False
python main.py mode=test_vllm
python main.py mode=test_vllm_multi_turn
```

Training automatically allocates `run_id` values such as `run-0001` under the method/model/data checkpoint path. `run_manifest.json` and the full `resolved_config.yaml` are written before preflight, tracking, and setup; `training_args.json` is added after argument instantiation. The manifest records `prepared`, `running`, `completed`, `failed`, or `interrupted` with the last `preflight`, `setup`, `training`, `saving`, or `completed` stage. Config values remain in `resolved_config.yaml`, effective trainer arguments remain in `training_args.json`, and the manifest keeps observed runtime facts and relative artifact references without duplicating either file. Runtime batch-size fields stay out of the checkpoint path. For distributed or multi-node runs, the manifest records planned and observed distributed, device, batch, and DataLoader runtime.

Single-turn test modes write the canonical `${connected_dir}/tests/${model_detail}/${dataset_name}.json` result and `${dataset_name}_manifest.json` companion. Multi-turn vLLM testing writes `${dataset_name}_multi_turn.jsonl` and `${dataset_name}_multi_turn_manifest.json` in the same model-detail directory. Companion manifests store the full resolved config once and add only the resolved test files, active data encoder, runtime backend, effective device map or tensor-parallel size, and actual generation parameters.

Decision uses HF forward scoring rather than generation. Its default namespace adds `/decision` below the model-detail directory, retains the canonical result/manifest names, and records `generation=null`. For a trained adapter, set `is_peft=True` and `peft_test.adapter_path`; the base-model example explicitly disables adapter loading.

MLflow is the default tracking backend. Use `tracking=wandb` for W&B or `tracking=mlflow_server` for the remote endpoint in `MLFLOW_TRACKING_URI`. The server profile leaves artifact location selection to the server and uses `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` when the endpoint requires MLflow basic authentication. Train runs write the selected backend and generated tracking UUID to `${output_dir}/tracking_metadata.json`; its location under the artifact run provides the mapping without copying config fields. Normal completion, ordinary exceptions, and `KeyboardInterrupt` or `SystemExit` end MLflow runs as `FINISHED`, `FAILED`, and `KILLED`, respectively.

## Script-based Execution

- `bash scripts/preprocessing/preprocess.sh`
- `bash scripts/preprocessing/preprocess_dataset.sh`
- `bash scripts/train/train.sh`
- `bash scripts/train/decision_train.sh`
- `bash scripts/train/rloo_train.sh`
- `bash scripts/train/async_grpo_train.sh`
- `bash scripts/postprocessing/merge_lora.sh`
- `bash scripts/test/test.sh`
- `bash scripts/test/test_large.sh`
- `bash scripts/test/decision_test.sh`
- `bash scripts/test/test_vllm.sh`
- `bash scripts/test/test_vllm_multi_turn.sh`

All methods default to `data_type=conversational`; select `data_type=structural` explicitly for separate input/target columns. Each dataset config has one target and accepts `data_type` as an argument; override fields directly through `dataset.<field>` or `test_dataset.<field>`.

For multi-node RLOO colocate training, configure the internal node address and GPU settings in `scripts/train/multinode/rloo_train_rank0.sh` and `rloo_train_rank1.sh`, then run the matching script on each node. No command-line experiment arguments are required. RLOO reuses the GRPO dataset contract and reward configuration; its default generation cadence follows `gradient_accumulation_steps`.

Decision multi-node launchers are `scripts/train/multinode/decision_train_rank0.sh` and `decision_train_rank1.sh`; configure their internal topology variables before use. Decision train/test profiles share `DecisionDataset` but remain separate; configure column names through `decision.columns`. See the Decision schema in README; do not feed ordinary SFT response rows without options and a target. Training/validation require hard or soft targets, while inference accepts target-free rows.

Async GRPO server flow:

Check the [Async GRPO attention kernel requirements](README.md#async-grpo-attention-kernel) before launch. For Blackwell GPUs, use `attn_implementation=flash_attention_2`, which is the default. Select `attn_implementation=kernels-community/flash-attn3` only on supported hardware. The vLLM server backend is configured independently; Hub access and offline startup conditions apply when selecting a Hub-backed kernel.

```bash
# script-managed server start/stop
# requirement: even number of GPUs and >=2 GPUs
bash scripts/train/async_grpo_train.sh
```

```bash
# config-only path (without train script)
# world_size=1: half GPUs for vLLM and half GPUs for trainer
# the trainer process restarts once before CUDA initialization with only its half visible
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
- SFT loss: `sft_loss_type={nll or dft}`; default `nll` is standard NLL, while `dft` weights it by the detached correct-token probability. Both use TRL's fused, tiled LM head. DFT does not support DeepSpeed sequence parallelism (Ulysses/ALST).
- SFT padding: `sft_padding_strategy=dynamic`; use `max_length` for fixed padding
- Decision: `decision.padding_strategy`, `decision.candidate_markers`, `decision.answer_prefix`, `decision.option_template`, `decision.columns`, `decision.ece_num_bins`, and inference-only `decision.inference_temperature`. Two candidates use BCE on their score difference; larger sets use CE. LoRA and dynamic padding are defaults; there is no text generation, new head, or automatic probability calibration. Overlong inputs are rejected without truncation.
- SFT Liger kernels: default `training_arguments.use_liger_kernel=False`. For the recommended Triton RoPE configuration, set `LIGER_KERNEL_IMPL_ROPE=triton` in `.env` before starting a new process and override `training_arguments.use_liger_kernel=True`. Use this configuration in environments affected by the CuTe RoPE/fork hang, and remove conflicting global CuTe overrides from `LIGER_KERNEL_IMPL` or `LIGER_KERNEL_BACKEND`. CuTe remains selectable in validated environments. Explicitly enabling `liger_kernel_config.fused_linear_cross_entropy` is unsupported.
- Training strategy: `strategy=deepspeed`; see [ZeRO-3 checkpoint resume](#zero-3-checkpoint-resume) for persistence tuning and full-state resume checks.
- Distributed launch: `distributed.enabled`, `distributed.num_machines`, `distributed.num_processes_per_machine`, `distributed.machine_rank`, `distributed.main_process_ip`, `distributed.main_process_port`
- Tracking backend: `tracking=wandb`, `tracking=mlflow`, `tracking=mlflow_server`
- PEFT/quantization: `is_quantized`, `is_peft`
- LoRA merge: `merge_max_shard_size`, `merge_pack_qwen_moe_experts`
- GRPO/vLLM: `use_vllm`, `vllm_mode`, `vllm_sync_strategy`
- RLOO: `num_generations`, `num_iterations`, `beta`, `epsilon`, `epsilon_high`, `steps_per_generation`; defaults are two generations, one iteration, left padding, and vLLM colocate with TP1. Keep the generation batch and global evaluation batch divisible by their generation counts. For VLM, use conversational prompts, keep `left_padding=True`, and set `decode_image_paths=True` for image-path datasets.
- VLM image augmentation: `image_augmentation.enabled`, `image_augmentation.probability`
- VLM image paths: `dataset_image.image_root_dir`, `dataset_image.convert_unsupported_extensions`; base64 images are decoded to PIL images when needed
- Reward extraction: `reward.extraction_profile`
- Retrieval rewards: `reward_embedding.preserved_env_keys`, `reward_embedding.isolated_env_keys`
- Model card/upload metadata: `upload_user`, `model_type`

## ZeRO-3 Checkpoint Resume

The shared ZeRO-3 config in `configs/deepspeed/ds_config.yaml` sets `zero_optimization.stage3_param_persistence_threshold: 1e5`, matching the [DeepSpeed default](https://deepspeed.readthedocs.io/en/stable/zero3.html#deepspeed.runtime.zero.config.DeepSpeedZeroConfig.param_persistence_threshold). The unit is the number of elements in one parameter tensor, not bytes or the model's total parameter count. Tensors at or below the threshold can remain replicated on each rank, subject to `stage3_model_persistence_threshold`. Lower values can reduce resident parameter memory but increase gather communication and CPU/GPU transfers with parameter offload; lower peak memory or faster training is not guaranteed.

PEFT LoRA + ZeRO-3 + reentrant checkpointing can raise `RuntimeError: tracing error` during full checkpoint resume. The observed trace expected a LoRA parameter but fetched a frozen base weight, rather than failing to read checkpoint files. With unmodified DeepSpeed 0.19.8 and PEFT 0.21.2, `1e6` reproduced this failure; `1e5` passed representative single-GPU and four-GPU checks with gradient accumulation 4. Upgrading DeepSpeed alone did not eliminate the failure at `1e6`. The lower threshold is a tested configuration, not a source fix or a guarantee for every model, LoRA rank, or offload setting.

Change the existing YAML setting or append this override to the original training command:

```shell
deepspeed.zero_optimization.stage3_param_persistence_threshold=1e5
```

LoRA A contains `r * in_features` elements: `r=128` with 4096 input features gives 524288 elements and is no longer persistent at this threshold, whereas `r=16` gives 65536 and may remain persistent. If another configuration fails, inspect the tensor sizes and persistence flags in the trace before tuning the threshold. Do not change unrelated offload, bucket, gradient-accumulation, or checkpointing settings merely to copy a diagnostic setup. ZeRO-2 does not use this setting.

Full resume restores the old optimizer and scheduler, unlike adapter continuation, which starts a new run. Apply these overrides to the original training command when restarting:

```shell
resume_training=True
resume_from_checkpoint=/path/to/run-0001/checkpoint-2
```

1. Create a short checkpoint with the intended model, LoRA, gradient accumulation, world size, and final DeepSpeed settings. Check that model/adapter, optimizer, scheduler, trainer, and per-rank RNG states were saved.
2. Restart in a new process with the original run settings and the resume overrides above. Keep the original checkpoint unchanged. If a path contains `=`, escape every `=` inside the value as `\=`; shell quoting alone does not escape Hydra syntax.
3. Confirm the same artifact `run_id` and tracking identity, check the effective DeepSpeed settings in `resolved_config.yaml` and `training_args.json`, and verify new optimizer steps beyond the restored `global_step`, new evaluation if enabled, checkpoint re-saving, and normal termination. Loading weights or replaying historical loss entries alone is not a resume pass. A successful resume from an older threshold's checkpoint does not replace checking a checkpoint created with the final settings.

## Validation Checklist

1. Mode command resolves Hydra config without errors.
2. Output artifacts are created for selected mode.
3. FlashAttention 2 and causal-conv1d install against the selected Torch and CUDA toolchain.
4. Changes are reflected in changelog and release notes before release.
