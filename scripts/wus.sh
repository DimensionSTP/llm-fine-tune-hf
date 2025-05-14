#!/bin/bash

path="src/scaling"
upload_user="Qwen"
model_type="Qwen3-8B"
precision="bf16"
wus_hidden_scale=2
num_safetensors=5

python $path/wus.py \
    upload_user=$upload_user \
    model_type=$model_type \
    precision=$precision \
    wus_hidden_scale=$wus_hidden_scale \
    num_safetensors=$num_safetensors
