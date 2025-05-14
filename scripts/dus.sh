#!/bin/bash

path="src/scaling"
upload_user="Qwen"
model_type="Qwen3-8B"
precision="bf16"
dus_hidden_layers=42
num_safetensors=1

python $path/dus.py \
    upload_user=$upload_user \
    model_type=$model_type \
    precision=$precision \
    dus_hidden_layers=$dus_hidden_layers \
    num_safetensors=$num_safetensors
