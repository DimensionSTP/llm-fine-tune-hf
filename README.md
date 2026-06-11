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

`chunked_nll` is SFT-only and reduces peak VRAM for long-context SFT while keeping the NLL objective. It is not compatible with `training_arguments.use_liger_kernel=True`.

* Use preprocessed tokenizer option

```shell
is_preprocessed={True or False}
```

* Dataset file path override

```shell
data_path=${connected_dir}/data
dataset_subdir={null or relative directory under data_path}
dataset_file_path={null or full dataset file path}
allow_dataset_file_name_mismatch={False or True}
```

`dataset_name` remains the logical logging/project/checkpoint identifier. By default the dataset file resolves to `${data_path}/${dataset_name}.${dataset_format}`. `dataset_subdir` changes only the directory, and `dataset_file_path` is an escape hatch; its basename must match `${dataset_name}.${dataset_format}` unless mismatch is explicitly allowed. Test data uses `test_dataset_subdir` and `test_dataset_file_path` with the same policy.

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
peft_initialization.adapter_name=adapter
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
image_augmentation.probability={0.0 to 1.0}
image_augmentation.rotation_degrees={degrees}
image_augmentation.jpeg_quality_min={1 to 100}
image_augmentation.jpeg_quality_max={1 to 100}
```

`image_augmentation` is disabled by default and applies only to training images for SFT, DPO, GRPO, async GRPO, and SDPO datasets. Validation, evaluation, and test datasets are not augmented. See `configs/image_augmentation/base.yaml` for all controls; `erase_area_min` and `erase_area_max` are area ratios. For bbox/grounding tasks, keep geometry-changing or evidence-removing options such as `rotation_degrees` and `erase_probability` disabled unless labels are transformed consistently.

VLM image paths are resolved through `dataset_image.image_root_dir` before they reach the processor. Relative paths are interpreted under that root, no-decode paths are normalized to absolute paths, and unsupported direct-path extensions such as `tif`/`tiff` are converted through PIL when `dataset_image.convert_unsupported_extensions=True`.

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

* Grounding bbox reward

```shell
reward.weight.grounding_bbox={0.0 or positive float}
reward.weight.grounding_selection={0.0 or positive float}
reward.grounding_bbox.category_token=ground
reward.grounding_selection.category_token=grounding
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
