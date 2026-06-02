#!/bin/bash

connected_dir="/data/llm-fine-tune-hf"
upload_user="Qwen"
model_type="Qwen3.5-9B"
model_detail=${model_type}
is_sft=True
is_quantized=False
is_peft=False
r=128
lora_alpha=512
max_length=4096
dataset_name="tulu"
strategy="deepspeed"
run_id="run-0001"

python -m src.postprocessing.merge_lora \
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
    dataset_name=$dataset_name \
    strategy=$strategy \
    run_id=$run_id