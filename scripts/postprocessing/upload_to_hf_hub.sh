#!/bin/bash

dataset_name="tulu"
strategy="deepspeed"
model_detail="Qwen3-8B"
upload_tag="sft"
is_sft=True
is_quantized=False
is_peft=False
max_length=4096
step=1000
run_id="run-0001"

python -m src.postprocessing.upload_to_hf_hub \
    dataset_name=$dataset_name \
    strategy=$strategy \
    model_detail=$model_detail \
    upload_tag=$upload_tag \
    is_sft=$is_sft \
    is_quantized=$is_quantized \
    is_peft=$is_peft \
    max_length=$max_length \
    run_id=$run_id \
    step=$step