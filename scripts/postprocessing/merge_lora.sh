#!/bin/bash

path="src/postprocessing"
connected_dir="/data/llm-fine-tune-hf"
upload_user="Qwen"
model_type="Qwen3-8B"
model_detail=${model_type}
is_sft=True
is_quantized=False
is_peft=False
r=128
lora_alpha=512
max_length=4096
batch_size=16
devices=8
gradient_accumulation_steps=1
dataset_name="tulu"
strategy="deepspeed"

python $path/merge_lora.py \
    connected_dir=$connected_dir \
    upload_user=$upload_user \
    model_type=$model_type \
    model_detail=$model_detail \
    is_sft=$is_sft \
    is_quantized=$is_quantized \
    is_peft=$is_peft \
    peft_config.r=$r \
    peft_config.lora_alpha=$lora_alpha \
    max_length=$max_length \
    batch_size=$batch_size \
    devices=$devices \
    gradient_accumulation_steps=$gradient_accumulation_steps \
    dataset_name=$dataset_name \
    strategy=$strategy
