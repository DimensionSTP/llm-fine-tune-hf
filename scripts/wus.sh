#!/bin/bash

path="src/scaling"
upload_user="HuggingFaceTB"
model_type="SmolLM2-360M-Instruct"
precision="bf16"
wus_hidden_scale=2
num_safetensors=1

python $path/wus.py \
    upload_user=$upload_user \
    model_type=$model_type \
    precision=$precision \
    wus_hidden_scale=$wus_hidden_scale \
    num_safetensors=$num_safetensors
