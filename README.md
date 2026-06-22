# LLM model fine-tuning & inference pipeline

## For (s)LLM model fine-tuning & inference

### Dataset

Any Structured Datasets

### Quick setup (conda)

```bash
# clone project
git clone https://github.com/DimensionSTP/llm-fine-tune-hf.git
cd llm-fine-tune-hf

# [OPTIONAL] create conda environment
conda create -n myenv python=3.12 -y
conda activate myenv

# install requirements
pip install --no-build-isolation -r requirements.txt
```

### Quick setup (pyproject.toml)

```bash
# install project dependencies from pyproject.toml
pip install --no-build-isolation .

# [OPTIONAL] editable install for development
pip install --no-build-isolation -e .
```

### Optional GPU dependency (flash-attn)

```bash
# Option A: install optional GPU extra from pyproject
pip install ".[gpu]"

# Option B: install directly from pinned Git commit
python -m pip install "flash-attn @ git+https://github.com/Dao-AILab/flash-attention.git@060c9188beec3a8b62b33a3bfa6d5d2d44975fab"
```

### Execution contract

- See `TRAINING_EVAL_CONTRACT.md` for required runtime inputs, supported modes, and output path expectations.

### .env file setting

```shell
PROJECT_DIR={PROJECT_DIR}
CONNECTED_DIR={CONNECTED_DIR}
DEVICES={DEVICES}
HF_HOME={HF_HOME}
USER_NAME={USER_NAME}
```

### Train

* end-to-end

```shell
python main.py mode=train
```

Training automatically allocates `run_id` values such as `run-0001` under the method/model/data checkpoint path and writes `run_manifest.json`, `resolved_config.yaml`, and `training_args.json` under `output_dir` before model construction. Runtime batch-size fields stay in metadata instead of the checkpoint path. For distributed or multi-node runs, set `distributed.enabled=true` and configure `distributed.num_machines`, `distributed.num_processes_per_machine`, `distributed.machine_rank`, `distributed.main_process_ip`, and `distributed.main_process_port`; `run_manifest.json` records planned and observed distributed, device, and batch runtime metadata. `run_metadata.allocation_timeout_seconds`, `run_metadata.allocation_poll_interval_seconds`, and `run_metadata.allocation_freshness_grace_seconds` control how non-rank0 processes wait for rank0's shared run directory allocation.

W&B is the default experiment tracking backend. Use `tracking=mlflow` to switch Trainer reporting and pipeline tracking helpers to MLflow. Train runs use a persisted `tracking_run_id` stored in `${output_dir}/tracking_metadata.json`; checkpoint `run_id` remains local to `output_base_dir`, and interrupted-run resume reuses the persisted tracking identity instead of falling back to `run_id` or starting a new backend run. MLflow stores its generated run UUID in the same metadata file and records artifact `run_id` as `artifact_run_id`.

### Test

* end-to-end

```shell
python main.py mode=test
```

* end-to-end(big model)

```shell
python main.py mode=test_large
```

* end-to-end(vLLM)

```shell
python main.py mode=test_vllm
```

* end-to-end(vLLM with multi-turn)

```shell
python main.py mode=test_vllm_multi_turn
```

`test_vllm` and `test_vllm_multi_turn` support VLM inputs by sending resolved images through vLLM `multi_modal_data` when `modality` is not `text`.

### Examples of shell scipts

* full preprocessing

```shell
bash scripts/preprocessing/preprocess.sh
```

* dataset preprocessing

```shell
bash scripts/preprocessing/preprocess_dataset.sh
```

* train

```shell
bash scripts/train/train.sh
```

* multi-node train

```shell
# Edit the variables at the top of both rank scripts before running:
# - main_process_ip: replace RANK0_NODE_IP with the rank 0 node address
# - machine_rank: 0 on rank 0, 1 on rank 1
# - gpu_ids: visible GPU ids on each node
# - model_type defaults to Qwen3.5-9B
#
# Run the matching rank scripts on separate nodes.
# On rank 1 node:
bash scripts/train/multinode/sft_train_rank1.sh

# On rank 0 node:
bash scripts/train/multinode/sft_train_rank0.sh
```

```shell
# GRPO external server mode uses one trainer node and one vLLM server node.
# Edit VLLM_SERVER_NODE_IP in the trainer script and vllm_model/server settings
# in the vLLM script before running.
# Dense-model vLLM server mode keeps data_parallel_size=1. The server script
# derives tensor_parallel_size from gpu_ids so all visible server GPUs are used
# through tensor parallelism.
# On vLLM server node:
bash scripts/train/multinode/grpo_server_vllm_rank1.sh

# On trainer node:
bash scripts/train/multinode/grpo_server_train_rank0.sh
```

* async GRPO train

```shell
# repo async runtime starts/stops vLLM server automatically
# requirement: even number of GPUs and >=2 GPUs
# async_runtime.vllm_server.tensor_parallel_size=auto maps to the GPU count
# assigned to the vLLM server side.
bash scripts/train/async_grpo_train.sh
```

```shell
# config-only path (without script)
# world_size=1: half GPUs for vLLM and half GPUs for trainer
# world_size=2: rank0=trainer, rank1=vLLM server
python main.py --config-name=async_grpo.yaml mode=train
```

```shell
# safe stop
# script/main runtime stops managed server automatically on exit
```

* test

```shell
bash scripts/test/test.sh
```

* test_large

```shell
bash scripts/test/test_large.sh
```

* test_vllm

```shell
bash scripts/test/test_vllm.sh
```

* test_vllm_multi_turn

```shell
bash scripts/test/test_vllm_multi_turn.sh
```

### Additional Options

* SFT train(masking input)

```shell
is_sft={True or False}
```

* SFT loss type

```shell
sft_loss_type={nll or chunked_nll}
```

`nll` is the default SFT loss type and is the supported default for assistant-only SFT (`is_sft=True`).

`chunked_nll` is an optional SFT-only loss path that reduces peak VRAM for long-context SFT while keeping the NLL objective. Use it for non-assistant-only long-context SFT when VRAM pressure is the bottleneck. Do not use it with `training_arguments.use_liger_kernel=True` or assistant-only SFT (`is_sft=True`). Smoke tests on both LLM and VLM paths showed that `chunked_nll + is_sft=True` can stall at the first training step. The likely reason is that `chunked_nll` drops `labels == -100` tokens before the LM head, while assistant-only masking makes valid label positions sparse and non-contiguous, causing the hidden-state gather/compaction path to become the bottleneck. `dynamic` padding is independent of the loss choice and is compatible with both `nll` and `chunked_nll`. The unsupported combination is only `chunked_nll + is_sft=True`.

* SFT padding strategy

```shell
sft_padding_strategy={max_length or dynamic}
```

`max_length` preserves the existing sample-level fixed padding path. `dynamic` keeps `max_length` as the truncation cap, pads each batch to the longest sample, uses `pad_to_multiple_of` when set, and is supported for both LLM and VLM SFT. SFT dynamic padding is right-padding only; `left_padding=True` fails fast.

* DataLoader runtime policy

```shell
dataloader_runtime.mode={auto or manual}
```

`auto` is the default. It resolves DataLoader workers per process from node-local CPU capacity and local rank count, then records the configured and resolved values in run metadata. The same resolved policy is used by Trainer-backed training and standalone test DataLoaders. All workloads use `prefetch_factor=2` when workers are enabled. `persistent_workers` is enabled only when the resolved worker count is greater than zero.

* Tracking backend

```shell
tracking={wandb or mlflow}
```

`wandb` is the default. `mlflow` requires the pinned MLflow dependency, uses `sqlite:///${connected_dir}/mlflow.db` and `file://${connected_dir}/mlflow-artifacts` by default, and writes `tracking_metadata.json` under each train `output_dir` so checkpoint artifact `run_id` can map to the MLflow run UUID on resume. Resume requires existing tracking metadata for both tracking backends.

* Supported training methods

```shell
python main.py --config-name={method}.yaml mode=train
```

| Method | Config | Dataset contract | Notes |
| --- | --- | --- | --- |
| SFT | `sft.yaml` | SFT | `nll` is the default loss type; `chunked_nll` is opt-in for non-assistant-only long-context SFT. |
| DPO | `dpo.yaml` | DPO | Preference-pair training. |
| KTO | `kto.yaml` | KTO | Unlikelihood-style preference training. |
| GKD | `gkd.yaml` | GKD | Distillation with `loss_type=nll`. |
| GRPO | `grpo.yaml` | GRPO | vLLM importance-sampling correction is enabled by default. |
| async GRPO | `async_grpo.yaml` | GRPO | Requires the trainer/vLLM split runtime. |
| SDPO | `sdpo.yaml` | GRPO-style reward | Uses the upstream TRL experimental trainer. |
| A2PO | `a2po.yaml` | GRPO-style reward | Uses the upstream TRL experimental trainer. |
| GOLD | `gold.yaml` | GKD-style teacher | Uses the upstream TRL experimental trainer. |

Liger remains opt-in through `training_arguments.use_liger_kernel=True` and is disabled by default.

* Use preprocessed tokenizer option

```shell
is_preprocessed={True or False}
```

* Dataset file path override

```shell
data_path=${connected_dir}/data
dataset_subdir={null or relative directory under data_path}
dataset_file_path={null or full dataset file path}
dataset_file_paths={null or list of dataset file paths}
dataset_files={null or list of {path, format, weight?} source specs}
val_dataset_file_path={null or full validation dataset file path}
val_dataset_file_paths={null or list of validation dataset file paths}
val_dataset_files={null or list of {path, format} validation source specs}
dataset_mix_name={null or explicit multi-dataset logging name}
allow_dataset_file_name_mismatch={False or True}
allow_val_dataset_file_name_mismatch={False or True}
test_dataset_file_path={null or full test dataset file path}
test_dataset_file_paths={null or list of test dataset file paths}
test_dataset_files={null or list of {path, format} test source specs}
dataset_resampling.enabled={False or True}
dataset_resampling.strategy={weighted_offline}
dataset_resampling.replacement={False or True}
dataset_resampling.target_size={null or positive integer}
```

`dataset_name` remains the logical dataset family. By default the train dataset resolves to `${data_path}/${dataset_name}.${dataset_format}`. `dataset_subdir` changes only the directory, `dataset_file_path` is a single-file escape hatch, and `dataset_file_paths` merges same-format files in order. `dataset_files` supports multi-format train sources and still merges by default; set `dataset_resampling.enabled=true` only when weighted offline resampling is required. If `dataset_file_paths` or `dataset_files` is set, `dataset_mix_name` is required and becomes `effective_dataset_name` for project, logging, and checkpoint naming. Explicit `val_dataset_file_path(s)` or `val_dataset_files` disables train-internal validation sampling and uses the provided validation source(s); validation and test sources support multi-format merge only, not weights or resampling. Test data uses `test_dataset_subdir`, `test_dataset_file_path`, `test_dataset_file_paths`, or `test_dataset_files` with the same basename mismatch policy. Default scripts do not need dataset override changes; scripted experiments should override primitive dataset keys and let composed names resolve from config.

* Left padding option

```shell
left_padding={True or False}
```

* Pure decoder based LLM QLoRA 4-bit quantization option

```shell
is_quantized={True or False}
```

* Pure decoder based LLM LoRA or QLoRA PEFT option

```shell
is_peft={True or False}
```

PEFT training uses fresh LoRA initialization by default:

```shell
peft_initialization.mode=fresh
```

To continue training an existing adapter without merging it into the base model, use:

```shell
is_peft=True
peft_initialization.mode=continue_from_adapter
peft_initialization.adapter_path=/path/to/adapter/checkpoint
peft_initialization.adapter_name=default
```

If the adapter path contains `=`, escape each `=` inside the value as `\=` and pass it as a Hydra-quoted value, such as `peft_initialization.adapter_path="/path/with\=a/checkpoint"`.

Adapter continuation is distinct from `resume_from_checkpoint`; it starts a new training run from an existing PEFT adapter, while `resume_from_checkpoint` resumes the same interrupted run. Continuation requires the adapter base model to match the current `pretrained_model_name`, disables merged-model auto-resolution, and incompatible adapter config or unsupported router-LoRA combinations fail fast. `async_grpo` adapter continuation is intentionally unsupported in this release.

* For LLM full fine-tuning(Continued Pretraining) in multi-GPU, recommended

```shell
strategy=deepspeed
```

`async_grpo` uses `strategy=none` for the trainer/vLLM split path.

* GRPO vLLM weight sync strategy

```shell
vllm_sync_strategy={default or lora_streaming}
```

* GRPO vLLM importance sampling

```shell
vllm_importance_sampling_correction={True or False}
vllm_importance_sampling_mode={sequence_mask}
vllm_importance_sampling_clip_min={null or float}
vllm_importance_sampling_clip_max={null or float}
vllm_importance_sampling_cap={null or float}
```

GRPO vLLM importance-sampling correction is enabled by default with `sequence_mask` mode, `clip_max=3.0`, and no min/cap.

* SDPO teacher server mode

```shell
teacher_model_kind={ema or live}
use_teacher_server={False or True}
vllm_mode={colocate or server}
vllm_server_base_url={null or http://host:port}
```

SDPO keeps `teacher_model_kind=ema`, `use_teacher_server=False`, and colocate vLLM by default. Teacher-server mode is opt-in and should be used with the upstream live-teacher server requirements: `teacher_model_kind=live`, `use_teacher_server=True`, and `vllm_mode=server`.

For dense-model external vLLM server mode, keep `data_parallel_size=1`.
Current vLLM offline server initialization rejects dense-model `data_parallel_size>1`,
so multi-GPU server utilization should be expressed with tensor parallelism.
This does not change colocate GRPO/SDPO defaults; colocate scripts keep
`vllm_tensor_parallel_size=1` unless the model itself requires tensor-parallel
loading inside the trainer-side vLLM instance.

`QLoRA + GRPO vLLM colocate` is not a supported release target. In colocate mode, TRL initializes vLLM from the quantized trainer model and vLLM stores bitsandbytes-packed weights, while the current LoRA streaming path sends dense LoRA-merged weights. Use non-quantized LoRA for colocate GRPO, or validate a separate external-server path before using QLoRA with vLLM.

* GRPO completion termination override

```shell
completion_termination.enabled={True or False}
completion_termination.terminal_token_texts=[</answer>]
completion_termination.terminal_token_ids=[]
completion_termination.infer_finished_from_short_completion={True or False}
completion_termination.include_model_generation_eos={True or False}
```

`completion_termination` is GRPO-only and disabled by default. It extends TRL's truncated-completion masking beyond tokenizer EOS/PAD by treating configured terminal token ids or token texts as valid completion terminators. `infer_finished_from_short_completion=True` also treats generations shorter than `max_completion_length` as terminated. SDPO and async GRPO do not use this option.

* VLM training image augmentation

```shell
image_augmentation.enabled={True or False}
image_augmentation.backend={pil or albumentations}
image_augmentation.probability={0.0 to 1.0}
image_augmentation.rotation_degrees={degrees}
image_augmentation.jpeg_quality_min={1 to 100}
image_augmentation.jpeg_quality_max={1 to 100}
```

`image_augmentation` is disabled by default and applies only to training images. Validation, evaluation, and test datasets are not augmented. The default backend is `pil`; `backend=albumentations` enables the extended VLM degradation stack and requires `albumentations` plus `opencv-python-headless`. See `configs/image_augmentation/base.yaml` for all controls; `erase_area_min` and `erase_area_max` are area ratios. For bbox/grounding tasks, keep geometry-changing or evidence-removing options such as `rotation_degrees`, `erase_probability`, `albumentations.resize.probability`, and `albumentations.coarse_dropout.probability` disabled unless labels are transformed consistently.

Example Albumentations smoke override:

```shell
image_augmentation.enabled=True
image_augmentation.backend=albumentations
image_augmentation.probability=1.0
image_augmentation.albumentations.seasoning.probability=1.0
```

For Qwen3-VL GRPO with colocated vLLM, small smoke runs may need an explicit `training_arguments.vllm_max_model_length` because the model advertises a very long context window and vLLM sizes KV cache from that value.

GRPO and SDPO expose `steps_per_generation` as an optional generation cadence control. The default `null` preserves TRL behavior, which resolves it to `gradient_accumulation_steps`. Lower values reduce the generation batch size and can reduce peak generation/buffer VRAM, but they usually increase generation frequency and may reduce throughput. Async GRPO does not expose this option in the current TRL API, and GOLD keeps its separate `generation_batch_size` contract because TRL's GOLDConfig does not expose `steps_per_generation`.

VLM image paths are resolved through `dataset_image.image_root_dir` before they reach the processor. Relative paths are interpreted under that root, no-decode paths are normalized to absolute paths, base64 images are decoded to PIL images when they would otherwise be misread as paths, and unsupported direct-path extensions such as `tif`/`tiff` are converted through PIL when `dataset_image.convert_unsupported_extensions=True`.

VLM dataset inputs use the image controls listed below where the selected dataset loader supports image fields. The default modality remains `text`.

| Dataset family | Image-capable | Image controls |
| --- | --- | --- |
| SFT | Yes | `modality`, `max_pixels`, `do_resize`, `image_augmentation`, `dataset_image` |
| DPO | Yes | `modality`, `max_pixels`, `do_resize`, `image_augmentation`, `decode_image_paths`, `dataset_image` |
| GRPO / async GRPO / SDPO / A2PO | Yes | `modality`, `max_pixels`, `do_resize`, `image_augmentation`, `decode_image_paths`, `dataset_image` |
| KTO | Yes | `modality`, `max_pixels`, `do_resize`, `image_augmentation`, `decode_image_paths`, `dataset_image` |
| GKD / GOLD | No | Text-only dataset contract |

* Reward embedding vLLM environment isolation

```shell
reward_embedding.preserved_env_keys=[RANK,WORLD_SIZE,LOCAL_RANK,CUDA_VISIBLE_DEVICES,MASTER_ADDR,MASTER_PORT,NCCL_SOCKET_IFNAME,NCCL_IB_DISABLE,VLLM_WORKER_MULTIPROC_METHOD]
reward_embedding.isolated_env_keys=[RANK,WORLD_SIZE,LOCAL_RANK]
```

* SFT response end template

```shell
response_end_template={null or template such as <|im_end|>}
```

`null` uses the tokenizer EOS token when masking SFT assistant labels.

SFT keeps the repo-owned torch Dataset masking path. `sft_label_mask.validation_mode=strict` fails fast when the response start template is missing, assistant labels are all `-100`, or prompt/padding tokens leak into labels; `report` keeps training unblocked for data inspection.

* Reward extraction profile

```shell
reward.extraction_profile={default or gemma4}
```

`default` keeps existing extraction behavior. `gemma4` strips Gemma channel/turn/tool stop markers before answer extraction. This affects rewards that call `extract_answer_from_generation()`; raw format rewards still check the original completion format.

Reward default hyperparameters and weights are centralized in `configs/reward/base.yaml`; reward class wiring stays in `configs/reward/manager.yaml`.

* KV reward

```shell
reward.weight.kv={0.0 or positive float}
```

`kv` is the only KV reward class for samples whose reward category contains the `kv` token, such as `single_kv` and `multi_kv`. It scores each sample once, so use `reward.weight.kv`.

* Grounding bbox reward

```shell
reward.weight.grounding_bbox={0.0 or positive float}
reward.weight.grounding_selection={0.0 or positive float}
reward.grounding_bbox.category_token=bbox
reward.grounding_selection.category_token=evidence
```

`grounding_bbox` is disabled by default. It is evaluated only when the sample reward category contains `reward.grounding_bbox.category_token`.

The label in `solution` should be a JSON object with `grounding_status`, optional `coord_system`, `positive_occurrences` for found targets, and optional `hard_negative_evidence`. Model answers should return JSON with `field_path`, `grounding_status`, and `evidence_occurrences`. Bounding boxes use `[x1, y1, x2, y2]` with page numbers in each occurrence or fragment. Schema aliases are controlled by `reward.grounding_bbox.schema_keys`.

For labels whose `grounding_status` is not `found`, the reward treats an answer with non-`found` status and empty `evidence_occurrences` as the correct negative grounding result.

`grounding_selection` is a separate candidate-selection reward for labels that expect top-level `grounding` lists rather than generated boxes. Each solution and prediction item is matched by `target_id`; item `selected_ids` are compared against the gold item, with the `selected_candidate_ids` alias supported for predictions. It does not use `grounding_status`; value targets are expected to provide evidence candidate ids through `selected_ids`. Missing gold targets, extra predicted targets, duplicate selected ids, wrong candidates, and over-selection are penalized without crashing malformed generations.

* Postprocessing artifact paths

```shell
bash scripts/postprocessing/merge_lora.sh
bash scripts/postprocessing/upload_to_hf_hub.sh
bash scripts/postprocessing/upload_all_to_hf_hub.sh
```

Postprocessing scripts keep `run_id` as a script-local variable. The Python entrypoint resolves `output_dir` from config-composed `output_base_dir` and `run_id`; scripts do not reconstruct checkpoint paths from batch size, devices, gradient accumulation, or timestamps.

* LoRA merge shard size

```shell
merge_max_shard_size={null or shard size such as 6GB}
```

* Pack unpacked Qwen MoE expert tensors after LoRA merge

```shell
merge_pack_qwen_moe_experts={True or False}
```

```shell
python -m src.postprocessing.merge_lora merge_max_shard_size=6GB merge_pack_qwen_moe_experts=True
```

Use `merge_pack_qwen_moe_experts=True` only for Qwen MoE checkpoints saved with unpacked per-expert tensors. Non-Qwen or already-packed checkpoints should keep the default `false`.

Packing rewrites safetensors shards through a temporary directory, so reserve enough disk space for another copy of the merged checkpoint.

* Upload user name and model name at HuggingFace Model card

```shell
upload_user={upload_user} 
model_type={model_type}
```

* Set data and target max length for model training and generation

```shell
max_length={max_length} 
```

__If you want to change main config, use --config-name={config_name}.__

__Also, you can use --multirun option.__

__You can set additional arguments through the command line.__
