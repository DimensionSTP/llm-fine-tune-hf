#!/bin/bash

num_machines=2
num_processes_per_machine=1
num_processes=$((num_machines * num_processes_per_machine))
machine_rank=0
main_process_ip="RANK0_NODE_IP"
main_process_port=29500
gpu_ids="0,1,2,3,4,5,6,7"

modality="text"
data_type="conversational"
split_ratio=1e-2
is_strict_split=False
dataset_name="tulu"
dataset_format="parquet"
is_preprocessed=False
strategy="none"
upload_user="Qwen"
model_type="Qwen3.5-9B"
revision="main"
left_padding=False
is_enable_thinking=False
is_quantized=False
is_peft=False
r=128
lora_alpha=512
target_modules="all-linear"
lora_dropout=0.0
max_length=2048
max_new_tokens=2048
is_bf16=True
batch_size=16
eval_batch_size=16
gradient_accumulation_steps=4
epsilon=3e-4
epsilon_high=4e-4
num_generations=16
temperature=0.7
log_completions=False
num_completions_to_print=3
vllm_server_host="VLLM_SERVER_NODE_IP"
vllm_server_port=8000
vllm_server_base_url="http://${vllm_server_host}:${vllm_server_port}"
vllm_server_timeout=240.0
request_timeout=600
max_inflight_tasks=-1
max_staleness=4
queue_maxsize=1024
weight_sync_steps=1
vllm_server_tensor_parallel_size=1
vllm_server_gpu_memory_utilization=0.95
vllm_server_log_path="/tmp/async_grpo_vllm_server.log"
vllm_server_ready_timeout=120
is_answer_tag=True
lr=5e-7
weight_decay=1e-1
warmup_ratio=5e-2
epoch=2
step=250
workers_ratio=8
use_all_workers=False

declare -A reward_weights=(
    ["think_format"]=0.0
    ["answer_format"]=0.0
    ["match"]=2.0
    ["code_execution"]=0.0
    ["rouge"]=0.0
    ["equation"]=0.0
    ["retrieval_hit"]=0.0
    ["retrieval_ndcg"]=0.0
    ["single_kv"]=0.0
    ["multi_kv"]=0.0
    ["grounding_bbox"]=0.0
)

reward_weight_keys=(
    "think_format"
    "answer_format"
    "match"
    "code_execution"
    "rouge"
    "equation"
    "retrieval_hit"
    "retrieval_ndcg"
    "single_kv"
    "multi_kv"
    "grounding_bbox"
)

reward_weight_args=()
for key in "${reward_weight_keys[@]}"; do
    reward_weight_args+=("reward.weight.${key}=${reward_weights[$key]}")
done

accelerate launch \
    --num_machines=$num_machines \
    --num_processes=$num_processes \
    --machine_rank=$machine_rank \
    --main_process_ip=$main_process_ip \
    --main_process_port=$main_process_port \
    --same_network \
    --gpu_ids=$gpu_ids \
    main.py --config-name=async_grpo.yaml mode=train \
    devices=$num_processes_per_machine \
    distributed.enabled=false \
    distributed.num_machines=$num_machines \
    distributed.num_processes_per_machine=$num_processes_per_machine \
    distributed.machine_rank=$machine_rank \
    distributed.main_process_ip=$main_process_ip \
    distributed.main_process_port=$main_process_port \
    distributed.validation_mode=error \
    modality=$modality \
    data_type=$data_type \
    split_ratio=$split_ratio \
    is_strict_split=$is_strict_split \
    dataset_name=$dataset_name \
    dataset_format=$dataset_format \
    is_preprocessed=$is_preprocessed \
    strategy=$strategy \
    upload_user=$upload_user \
    model_type=$model_type \
    revision=$revision \
    left_padding=$left_padding \
    is_enable_thinking=$is_enable_thinking \
    is_quantized=$is_quantized \
    is_peft=$is_peft \
    peft_config.r=$r \
    peft_config.lora_alpha=$lora_alpha \
    peft_config.target_modules=$target_modules \
    peft_config.lora_dropout=$lora_dropout \
    max_length=$max_length \
    max_new_tokens=$max_new_tokens \
    is_bf16=$is_bf16 \
    batch_size=$batch_size \
    eval_batch_size=$eval_batch_size \
    gradient_accumulation_steps=$gradient_accumulation_steps \
    epsilon=$epsilon \
    epsilon_high=$epsilon_high \
    num_generations=$num_generations \
    generation_config.temperature=$temperature \
    log_completions=$log_completions \
    num_completions_to_print=$num_completions_to_print \
    vllm_server_base_url=$vllm_server_base_url \
    vllm_server_timeout=$vllm_server_timeout \
    request_timeout=$request_timeout \
    max_inflight_tasks=$max_inflight_tasks \
    max_staleness=$max_staleness \
    queue_maxsize=$queue_maxsize \
    weight_sync_steps=$weight_sync_steps \
    async_runtime.enable=true \
    async_runtime.vllm_server.auto_start=true \
    async_runtime.vllm_server.auto_stop=true \
    async_runtime.vllm_server.host=$vllm_server_host \
    async_runtime.vllm_server.port=$vllm_server_port \
    async_runtime.vllm_server.ready_timeout=$vllm_server_ready_timeout \
    async_runtime.vllm_server.tensor_parallel_size=$vllm_server_tensor_parallel_size \
    async_runtime.vllm_server.gpu_memory_utilization=$vllm_server_gpu_memory_utilization \
    async_runtime.vllm_server.log_path=$vllm_server_log_path \
    reward.is_answer_tag=$is_answer_tag \
    "${reward_weight_args[@]}" \
    lr=$lr \
    weight_decay=$weight_decay \
    warmup_ratio=$warmup_ratio \
    epoch=$epoch \
    step=$step \
    workers_ratio=$workers_ratio \
    use_all_workers=$use_all_workers
