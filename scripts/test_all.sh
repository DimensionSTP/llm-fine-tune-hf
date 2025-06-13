#!/bin/bash

data_type="conversational"

dataset_names=(
    "aime24"
    "arc_challenge"
    "arc_easy"
    "gpqa_diamond"
    "gpqa_extended"
    "gpqa_main"
    "gsm8k"
    "humaneval"
    "math500"
    "mbpp"
    "mmlu"
    "mmlu_pro"
    "open_rewrite_eval"
    "hrm8k_gsm8k"
    "hrm8k_ksm"
    "hrm8k_math"
    "hrm8k_mmmlu"
    "hrm8k_omni_math"
    "kmmlu"
    "haerae"
)

is_preprocessed=False
upload_user="Qwen"

model_types=(
    "Qwen3-0.6B"
    "Qwen3-1.7B"
    "Qwen3-4B"
    "Qwen3-8B"
    "Qwen3-14B"
    "Qwen3-32B"
    "Qwen3-30B-A3B"
    "Qwen3-235B-A22B"
)

left_padding=True
max_length=2048
max_new_tokens=256
eval_batch_size=16
workers_ratio=8
use_all_workers=False
num_gpus=$(nvidia-smi -L | wc -l)

for model_type in "${model_types[@]}"
do
    for dataset_name in "${dataset_names[@]}"
    do
        echo "Running evaluation on dataset: $dataset_name"

        torchrun --nproc_per_node=$num_gpus main.py mode=test \
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
    done
done
