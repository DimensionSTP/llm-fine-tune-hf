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
max_length=4096
max_new_tokens=512
do_sample=True
temperature=0.6
top_p=0.95
top_k=20
eval_batch_size=16
workers_ratio=8
use_all_workers=False
num_gpus=$(nvidia-smi -L | wc -l)

torchrun --nproc_per_node=$num_gpus main.py mode=test \
    data_type=$data_type \
    dataset_name=$dataset_name \
    dataset_format=$dataset_format \
    is_preprocessed=$is_preprocessed \
    upload_user=$upload_user \
    model_type=$model_type \
    revision=$revision \
    left_padding=$left_padding \
    is_enable_thinking=$is_enable_thinking \
    max_length=$max_length \
    max_new_tokens=$max_new_tokens \
    do_sample=$do_sample \
    generation_config.temperature=$temperature \
    generation_config.top_p=$top_p \
    generation_config.top_k=$top_k \
    eval_batch_size=$eval_batch_size \
    workers_ratio=$workers_ratio \
    use_all_workers=$use_all_workers


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
max_length=4096
max_new_tokens=512
do_sample=True
temperature=0.6
top_p=0.95
top_k=20
eval_batch_size=16
workers_ratio=8
use_all_workers=False
num_gpus=$(nvidia-smi -L | wc -l)

torchrun --nproc_per_node=$num_gpus main.py mode=test \
    data_type=$data_type \
    dataset_name=$dataset_name \
    dataset_format=$dataset_format \
    is_preprocessed=$is_preprocessed \
    upload_user=$upload_user \
    model_type=$model_type \
    revision=$revision \
    left_padding=$left_padding \
    is_enable_thinking=$is_enable_thinking \
    max_length=$max_length \
    max_new_tokens=$max_new_tokens \
    do_sample=$do_sample \
    generation_config.temperature=$temperature \
    generation_config.top_p=$top_p \
    generation_config.top_k=$top_k \
    eval_batch_size=$eval_batch_size \
    workers_ratio=$workers_ratio \
    use_all_workers=$use_all_workers


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
r=32
lora_alpha=64
peft_detail="r\=${r}-a\=${lora_alpha}"
adapter_path="${base_path}/${train_dataset}/${model_type}/${peft_detail}"
left_padding=True
is_enable_thinking=False
max_length=4096
max_new_tokens=512
do_sample=True
temperature=0.6
top_p=0.95
top_k=20
eval_batch_size=16
test_output_dir="${base_path}/tests/${train_dataset}/${model_type}/${peft_detail}"
workers_ratio=8
use_all_workers=False
num_gpus=$(nvidia-smi -L | wc -l)

torchrun --nproc_per_node=$num_gpus main.py mode=test \
    data_type=$data_type \
    dataset_name=$dataset_name \
    dataset_format=$dataset_format \
    is_preprocessed=$is_preprocessed \
    upload_user=$upload_user \
    model_type=$model_type \
    revision=$revision \
    peft_test.adapter_path=$adapter_path \
    left_padding=$left_padding \
    is_enable_thinking=$is_enable_thinking \
    max_length=$max_length \
    max_new_tokens=$max_new_tokens \
    do_sample=$do_sample \
    generation_config.temperature=$temperature \
    generation_config.top_p=$top_p \
    generation_config.top_k=$top_k \
    eval_batch_size=$eval_batch_size \
    test_output_dir=$test_output_dir \
    workers_ratio=$workers_ratio \
    use_all_workers=$use_all_workers