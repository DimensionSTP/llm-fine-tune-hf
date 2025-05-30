#!/bin/bash

path="src/postprocessing"
dataset_name="tulu"
is_sft=False
is_preprocessed=False
strategy="deepspeed"
upload_user="Qwen"
model_type="Qwen3-8B"
model_detail="Qwen3-8B"
upload_tag="sft"
is_quantized=False
is_peft=False
max_length=4096
batch_size=16
gradient_accumulation_steps=1

python $path/upload_all_to_hf_hub.py \
    dataset_name=$dataset_name \
    is_sft=$is_sft \
    is_preprocessed=$is_preprocessed \
    strategy=$strategy \
    upload_user=$upload_user \
    model_type=$model_type \
    model_detail=$model_detail \
    upload_tag=$upload_tag \
    is_quantized=$is_quantized \
    is_peft=$is_peft \
    max_length=$max_length \
    batch_size=$batch_size \
    gradient_accumulation_steps=$gradient_accumulation_steps
