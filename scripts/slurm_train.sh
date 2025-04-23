#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH --partition=8gpu
#SBATCH --gres=gpu:8
#SBATCH --nodelist=gpu-8-003
#SBATCH --output=logs/train_output.log
#SBATCH --error=logs/train_error.log

cd ~/llm-fine-tune

module add compilers/cuda/12.4 compilers/gcc/10.2.0 libraries/nccl/2.21.5
source activate myenv

data_type="structural"
split_ratio=1e-4
dataset_name="tulu"
is_sft=False
is_preprocessed=False
strategy="deepspeed"
upload_user="HuggingFaceTB"
model_type="SmolLM2-135M-Instruct"
left_padding=False
model_detail="SmolLM2-135M-135M-Instruct"
is_quantized=False
is_peft=False
max_length=4096
is_bf16=True
batch_size=16
eval_batch_size=16
gradient_accumulation_steps=8
lr=3e-5
weight_decay=1e-1
warmup_ratio=5e-2
epoch=5
step=100
workers_ratio=8
use_all_workers=False

if [ "$strategy" = "deepspeed" ]; then
    deepspeed main.py mode=train \
        data_type=$data_type \
        split_ratio=$split_ratio \
        dataset_name=$dataset_name \
        is_sft=$is_sft \
        is_preprocessed=$is_preprocessed \
        strategy=$strategy \
        upload_user=$upload_user \
        model_type=$model_type \
        left_padding=$left_padding \
        model_detail=$model_detail \
        is_quantized=$is_quantized \
        is_peft=$is_peft \
        max_length=$max_length \
        is_bf16=$is_bf16 \
        batch_size=$batch_size \
        eval_batch_size=$eval_batch_size \
        gradient_accumulation_steps=$gradient_accumulation_steps \
        lr=$lr \
        weight_decay=$weight_decay \
        warmup_ratio=$warmup_ratio \
        epoch=$epoch \
        step=$step \
        workers_ratio=$workers_ratio \
        use_all_workers=$use_all_workers
else
    python main.py mode=train \
        data_type=$data_type \
        split_ratio=$split_ratio \
        dataset_name=$dataset_name \
        is_sft=$is_sft \
        is_preprocessed=$is_preprocessed \
        strategy=$strategy \
        upload_user=$upload_user \
        model_type=$model_type \
        left_padding=$left_padding \
        model_detail=$model_detail \
        is_quantized=$is_quantized \
        is_peft=$is_peft \
        max_length=$max_length \
        is_bf16=$is_bf16 \
        batch_size=$batch_size \
        eval_batch_size=$eval_batch_size \
        gradient_accumulation_steps=$gradient_accumulation_steps \
        lr=$lr \
        weight_decay=$weight_decay \
        warmup_ratio=$warmup_ratio \
        epoch=$epoch \
        step=$step \
        workers_ratio=$workers_ratio \
        use_all_workers=$use_all_workers
fi
