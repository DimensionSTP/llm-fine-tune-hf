#!/bin/bash

num_machines=2
num_processes_per_machine=8
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
strategy="deepspeed"
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
beta=0.0
epsilon=0.2
epsilon_high=null
num_generations=16
temperature=0.7
top_p=0.8
top_k=20
importance_sampling_level=token
loss_type=dapo
scale_rewards=group
use_vllm=True
vllm_mode=colocate
vllm_tensor_parallel_size=1
gpu_memory_utilization=0.7
distillation_weight=1.0
teacher_model_kind=ema
teacher_update_rate=0.05
teacher_sync_steps=1
use_successful_as_teacher=True
success_reward_threshold=1.0
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
    ["grounding_selection"]=0.0
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
    "grounding_selection"
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
    main.py --config-name=sdpo.yaml mode=train \
    devices=$num_processes_per_machine \
    distributed.enabled=true \
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
    beta=$beta \
    epsilon=$epsilon \
    epsilon_high=$epsilon_high \
    num_generations=$num_generations \
    generation_config.temperature=$temperature \
    generation_config.top_p=$top_p \
    generation_config.top_k=$top_k \
    importance_sampling_level=$importance_sampling_level \
    loss_type=$loss_type \
    scale_rewards=$scale_rewards \
    use_vllm=$use_vllm \
    vllm_mode=$vllm_mode \
    vllm_tensor_parallel_size=$vllm_tensor_parallel_size \
    gpu_memory_utilization=$gpu_memory_utilization \
    distillation_weight=$distillation_weight \
    teacher_model_kind=$teacher_model_kind \
    teacher_update_rate=$teacher_update_rate \
    teacher_sync_steps=$teacher_sync_steps \
    use_successful_as_teacher=$use_successful_as_teacher \
    success_reward_threshold=$success_reward_threshold \
    reward.is_answer_tag=$is_answer_tag \
    "${reward_weight_args[@]}" \
    lr=$lr \
    weight_decay=$weight_decay \
    warmup_ratio=$warmup_ratio \
    epoch=$epoch \
    step=$step \
    workers_ratio=$workers_ratio \
    use_all_workers=$use_all_workers