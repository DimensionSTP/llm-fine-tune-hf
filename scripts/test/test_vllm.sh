#!/bin/bash

# base
data_type="conversational"
dataset_name="mmlu"
dataset_format="parquet"
is_preprocessed=False
upload_user="Qwen"
model_type="Qwen3-8B"
revision="main"
left_padding=True
is_enable_thinking=False
is_quantized=False
is_peft=False
max_length=4096
max_new_tokens=512
do_sample=True
temperature=0.6
top_p=0.95
top_k=20
gpu_memory_utilization=0.95

python main.py mode=test_vllm \
    data_type=$data_type \
    dataset_name=$dataset_name \
    dataset_format=$dataset_format \
    is_preprocessed=$is_preprocessed \
    upload_user=$upload_user \
    model_type=$model_type \
    revision=$revision \
    left_padding=$left_padding \
    is_enable_thinking=$is_enable_thinking \
    is_quantized=$is_quantized \
    is_peft=$is_peft \
    max_length=$max_length \
    max_new_tokens=$max_new_tokens \
    do_sample=$do_sample \
    generation_config.temperature=$temperature \
    generation_config.top_p=$top_p \
    generation_config.top_k=$top_k \
    gpu_memory_utilization=$gpu_memory_utilization


# fine-tuned
data_type="conversational"
dataset_name="mmlu"
dataset_format="parquet"
is_preprocessed=False
upload_user="/data/llm-fine-tune-hf"
train_dataset="tulu"
model_type="${train_dataset}/Qwen3-8B"
revision="main"
left_padding=True
is_enable_thinking=False
is_quantized=False
is_peft=False
max_length=4096
max_new_tokens=512
do_sample=True
temperature=0.6
top_p=0.95
top_k=20
gpu_memory_utilization=0.95

python main.py mode=test_vllm \
    data_type=$data_type \
    dataset_name=$dataset_name \
    dataset_format=$dataset_format \
    is_preprocessed=$is_preprocessed \
    upload_user=$upload_user \
    model_type=$model_type \
    revision=$revision \
    left_padding=$left_padding \
    is_enable_thinking=$is_enable_thinking \
    is_quantized=$is_quantized \
    is_peft=$is_peft \
    max_length=$max_length \
    max_new_tokens=$max_new_tokens \
    do_sample=$do_sample \
    generation_config.temperature=$temperature \
    generation_config.top_p=$top_p \
    generation_config.top_k=$top_k \
    gpu_memory_utilization=$gpu_memory_utilization


# LoRA fine-tuned
data_type="conversational"
dataset_name="mmlu"
dataset_format="parquet"
is_preprocessed=False
upload_user="Qwen"
base_path="/data/llm-fine-tune-hf"
train_dataset="tulu"
model_type="Qwen3-8B"
revision="main"
left_padding=True
is_enable_thinking=False
is_quantized=False
is_peft=True
r=32
lora_alpha=64
peft_detail="r\=${r}-a\=${lora_alpha}"
model_detail="${train_dataset}/${model_type}/${peft_detail}"
adapter_path="${base_path}/${model_detail}"
max_length=4096
max_new_tokens=512
do_sample=True
temperature=0.6
top_p=0.95
top_k=20
gpu_memory_utilization=0.95
test_output_dir="${base_path}/tests/${model_detail}"

python main.py mode=test_vllm \
    data_type=$data_type \
    dataset_name=$dataset_name \
    dataset_format=$dataset_format \
    is_preprocessed=$is_preprocessed \
    upload_user=$upload_user \
    model_type=$model_type \
    revision=$revision \
    left_padding=$left_padding \
    is_enable_thinking=$is_enable_thinking \
    is_quantized=$is_quantized \
    is_peft=$is_peft \
    peft_config.r=$r \
    peft_config.lora_alpha=$lora_alpha \
    model_detail=$model_detail \
    peft_test.adapter_path=$adapter_path \
    max_length=$max_length \
    max_new_tokens=$max_new_tokens \
    do_sample=$do_sample \
    generation_config.temperature=$temperature \
    generation_config.top_p=$top_p \
    generation_config.top_k=$top_k \
    gpu_memory_utilization=$gpu_memory_utilization \
    test_output_dir=$test_output_dir