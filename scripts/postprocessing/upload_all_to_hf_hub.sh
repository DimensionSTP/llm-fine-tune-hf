#!/bin/bash

path="src/postprocessing"
dataset_name="tulu"
strategy="deepspeed"
model_detail="Qwen3-8B"
upload_tag="sft"
is_sft=True
is_quantized=False
is_peft=False
max_length=4096
batch_size=16
gradient_accumulation_steps=1

python $path/upload_all_to_hf_hub.py \
    dataset_name=$dataset_name \
    strategy=$strategy \
    model_detail=$model_detail \
    upload_tag=$upload_tag \
    is_sft=$is_sft \
    is_quantized=$is_quantized \
    is_peft=$is_peft \
    max_length=$max_length \
    batch_size=$batch_size \
    gradient_accumulation_steps=$gradient_accumulation_steps
