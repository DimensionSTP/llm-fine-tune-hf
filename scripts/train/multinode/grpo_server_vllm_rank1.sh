#!/bin/bash

gpu_ids="0,1,2,3,4,5,6,7"
vllm_model="Qwen/Qwen3.5-9B"
vllm_revision="main"
vllm_host="0.0.0.0"
vllm_port=8000
IFS="," read -ra visible_gpu_ids <<< "$gpu_ids"
vllm_tensor_parallel_size=${#visible_gpu_ids[@]}
vllm_data_parallel_size=1
vllm_gpu_memory_utilization=0.95
vllm_dtype="bfloat16"
vllm_max_model_len=4096
vllm_enforce_eager=True

CUDA_VISIBLE_DEVICES=$gpu_ids python -m trl.scripts.vllm_serve \
    --model "$vllm_model" \
    --revision "$vllm_revision" \
    --tensor_parallel_size "$vllm_tensor_parallel_size" \
    --data_parallel_size "$vllm_data_parallel_size" \
    --host "$vllm_host" \
    --port "$vllm_port" \
    --gpu_memory_utilization "$vllm_gpu_memory_utilization" \
    --dtype "$vllm_dtype" \
    --max_model_len "$vllm_max_model_len" \
    --enforce_eager "$vllm_enforce_eager" \
    --trust_remote_code True \
    --log_level info
