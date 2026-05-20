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
pip install -r requirements.txt
```

### Quick setup (pyproject.toml)

```bash
# install project dependencies from pyproject.toml
pip install .

# [OPTIONAL] editable install for development
pip install -e .
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

* LoRA merge shard size

```shell
merge_max_shard_size={null or shard size such as 6GB}
```

* Pack unpacked Qwen MoE expert tensors after LoRA merge

```shell
merge_pack_qwen_moe_experts={True or False}
```

```shell
python src/postprocessing/merge_lora.py merge_max_shard_size=6GB merge_pack_qwen_moe_experts=True
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
