#!/bin/bash

data_type="conversational"
dataset_name="mmlu"
is_preprocessed=False
upload_user="Qwen"
model_type="Qwen3-8B"
revision="main"
left_padding=True
is_enable_thinking=False
max_length=2048
max_new_tokens=256
do_sample=True
temperature=0.6
top_p=0.95
top_k=20
eval_batch_size=16
workers_ratio=8
use_all_workers=False

python main.py mode=test_large \
    data_type=$data_type \
    dataset_name=$dataset_name \
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
