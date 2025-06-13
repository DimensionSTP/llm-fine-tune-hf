#!/bin/bash

data_type="conversational"
dataset_name="mmlu"
is_preprocessed=False
upload_user="Qwen"
model_type="Qwen3-235B-A22B"
left_padding=True
max_length=2048
max_new_tokens=256
eval_batch_size=16
workers_ratio=8
use_all_workers=False

python main.py mode=test_large \
    data_type=$data_type \
    dataset_name=$dataset_name \
    is_preprocessed=$is_preprocessed \
    upload_user=$upload_user \
    model_type=$model_type \
    left_padding=$left_padding \
    max_length=$max_length \
    max_new_tokens=$max_new_tokens \
    eval_batch_size=$eval_batch_size \
    workers_ratio=$workers_ratio \
    use_all_workers=$use_all_workers
