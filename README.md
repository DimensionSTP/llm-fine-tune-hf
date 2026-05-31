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

Training automatically allocates `run_id` values such as `run-0001` under the method/model/data checkpoint path and writes `run_manifest.json`, `resolved_config.yaml`, and `training_args.json` under `output_dir` before model construction. Runtime batch-size fields stay in metadata instead of the checkpoint path.

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

* async GRPO train

```shell
# script starts/stops vLLM server automatically
# requirement: even number of GPUs and >=2 GPUs
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
# 1) script/main runtime stops managed server automatically on exit
# 2) fallback manual cleanup for script-managed server
if [ -f /tmp/async_grpo_vllm_server.pid ]; then kill "$(cat /tmp/async_grpo_vllm_server.pid)"; fi
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

* For LLM full fine-tuning(Continued Pretraining) in multi-GPU, recommended

```shell
strategy=deepspeed
```

* GRPO vLLM weight sync strategy

```shell
vllm_sync_strategy={default or lora_streaming}
```

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

* Reward extraction profile

```shell
reward.extraction_profile={default or gemma4}
```

`default` keeps existing extraction behavior. `gemma4` strips Gemma channel/turn/tool stop markers before answer extraction. This affects rewards that call `extract_answer_from_generation()`; raw format rewards still check the original completion format.

* Grounding bbox reward

```shell
reward.weight.grounding_bbox={0.0 or positive float}
reward.grounding_bbox.category_token=ground
```

`grounding_bbox` is disabled by default. It is evaluated only when the sample reward category contains `reward.grounding_bbox.category_token`.

The label in `solution` should be a JSON object with `grounding_status`, optional `coord_system`, `positive_occurrences` for found targets, and optional `hard_negative_evidence`. Model answers should return JSON with `field_path`, `grounding_status`, and `evidence_occurrences`. Bounding boxes use `[x1, y1, x2, y2]` with page numbers in each occurrence or fragment.

For labels whose `grounding_status` is not `found`, the reward treats an answer with non-`found` status and empty `evidence_occurrences` as the correct negative grounding result.

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
