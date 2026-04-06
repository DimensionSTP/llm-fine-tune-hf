#!/bin/bash

set -euo pipefail

modality="text"
data_type="conversational"
split_ratio=1e-2
is_strict_split=False
dataset_name="tulu"
dataset_format="parquet"
is_preprocessed=False
strategy="deepspeed"
upload_user="Qwen"
model_type="Qwen3-8B"
revision="main"
left_padding=False
is_enable_thinking=False
is_quantized=False
is_peft=False
r=128
lora_alpha=256
target_modules="all-linear"
lora_dropout=0.0
max_length=2048
max_new_tokens=2048
is_bf16=True
batch_size=16
eval_batch_size=16
gradient_accumulation_steps=4
epsilon=3e-4
epsilon_high=4e-4
num_generations=16
temperature=0.7
log_completions=False
num_completions_to_print=3
vllm_server_host="127.0.0.1"
vllm_server_port=8000
vllm_server_base_url="http://${vllm_server_host}:${vllm_server_port}"
vllm_server_timeout=240.0
request_timeout=600
max_inflight_tasks=-1
max_staleness=4
queue_maxsize=1024
weight_sync_steps=1
vllm_server_model="${VLLM_SERVER_MODEL:-Qwen/Qwen3-8B}"
vllm_server_tensor_parallel_size=1
vllm_server_gpu_memory_utilization="${VLLM_SERVER_GPU_MEMORY_UTILIZATION:-0.85}"
vllm_server_log_path="${VLLM_SERVER_LOG_PATH:-/tmp/async_grpo_vllm_server.log}"
vllm_server_pid_path="${VLLM_SERVER_PID_PATH:-/tmp/async_grpo_vllm_server.pid}"
is_answer_tag=True
lr=5e-7
weight_decay=1e-1
warmup_ratio=5e-2
epoch=2
step=250
workers_ratio=8
use_all_workers=False
vllm_gpu_ids=""
trainer_gpu_ids=""

declare -A reward_weights=(
    ["think_format"]=0.0
    ["answer_format"]=0.0
    ["match"]=2.0
    ["code_execution"]=0.0
    ["rouge"]=0.0
    ["equation"]=0.0
    ["retrieval_hit"]=0.0
    ["retrieval_ndcg"]=0.0
)

reward_weight_keys=(
    "think_format"
    "answer_format"
    "match"
    "code_execution"
    "rouge"
    "equation"
    "retrieval_hit"
    "retrieval_ndcg"
)

reward_weight_args=()
for key in "${reward_weight_keys[@]}"; do
    reward_weight_args+=("reward.weight.${key}=${reward_weights[$key]}")
done

resolve_half_gpu_partition() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[error] async_grpo requires nvidia-smi for half-half GPU partition."
        exit 1
    fi

    local physical_count
    physical_count="$(nvidia-smi -L 2>/dev/null | grep -Ec '^GPU [0-9]+:' || true)"
    if [ -z "${physical_count}" ] || [ "${physical_count}" -lt 2 ]; then
        echo "[error] async_grpo requires >=2 GPUs. detected=${physical_count:-0}"
        exit 1
    fi

    if [ $((physical_count % 2)) -ne 0 ]; then
        echo "[error] async_grpo half-half partition requires even GPU count. detected=${physical_count}"
        exit 1
    fi

    local split_idx
    split_idx=$((physical_count / 2))

    vllm_gpu_ids="$(seq -s, 0 $((split_idx - 1)))"
    trainer_gpu_ids="$(seq -s, "${split_idx}" $((physical_count - 1)))"

    echo "[async_grpo] GPU partition (half-half): vllm=${vllm_gpu_ids}, trainer=${trainer_gpu_ids}, physical=${physical_count}"
}

wait_vllm_server_ready() {
    local base_url="$1"
    local max_wait_sec="${2:-120}"

    for _ in $(seq 1 "$max_wait_sec"); do
        if python - "$base_url" <<'PY'
import json
import sys
import urllib.request

base_url = sys.argv[1]
url = base_url + "/v1/models"
try:
    with urllib.request.urlopen(url, timeout=1.0) as response:
        if response.status == 200:
            payload = json.loads(response.read().decode("utf-8"))
            if "data" in payload:
                raise SystemExit(0)
except Exception:
    pass
raise SystemExit(1)
PY
        then
            return 0
        fi
        sleep 1
    done
    return 1
}

start_vllm_server() {
    echo "[async_grpo] Starting vLLM server at ${vllm_server_base_url}"
    CUDA_VISIBLE_DEVICES="${vllm_gpu_ids}" VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}" python -m vllm.entrypoints.openai.api_server \
        --model "${vllm_server_model}" \
        --host "${vllm_server_host}" \
        --port "${vllm_server_port}" \
        --tensor-parallel-size "${vllm_server_tensor_parallel_size}" \
        --gpu-memory-utilization "${vllm_server_gpu_memory_utilization}" \
        --weight-transfer-config '{"backend":"nccl"}' \
        > "${vllm_server_log_path}" 2>&1 &
    local pid=$!
    echo "${pid}" > "${vllm_server_pid_path}"
}

cleanup_vllm_server() {
    if [ ! -f "${vllm_server_pid_path}" ]; then
        return
    fi

    local pid
    pid="$(cat "${vllm_server_pid_path}")"
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
        echo "[async_grpo] Stopping vLLM server pid=${pid}"
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    fi
    rm -f "${vllm_server_pid_path}"
}

if wait_vllm_server_ready "${vllm_server_base_url}" 1; then
    echo "[error] Existing vLLM server already running at ${vllm_server_base_url}"
    echo "[hint] Stop existing server or change port, then rerun."
    exit 1
fi

resolve_half_gpu_partition
start_vllm_server
trap cleanup_vllm_server EXIT

if ! wait_vllm_server_ready "${vllm_server_base_url}" 120; then
    echo "[error] vLLM server did not become ready: ${vllm_server_base_url}"
    echo "[hint] See log: ${vllm_server_log_path}"
    cleanup_vllm_server
    exit 1
fi

CUDA_VISIBLE_DEVICES="${trainer_gpu_ids}" accelerate launch main.py --config-name=async_grpo.yaml mode=train \
    modality=$modality \
    data_type=$data_type \
    split_ratio=$split_ratio \
    is_strict_split=$is_strict_split \
    dataset_name=$dataset_name \
    dataset_format=$dataset_format \
    is_preprocessed=$is_preprocessed \
    strategy=$strategy \
    upload_user=$upload_user \
    model_type=$model_type \
    revision=$revision \
    left_padding=$left_padding \
    is_enable_thinking=$is_enable_thinking \
    is_quantized=$is_quantized \
    is_peft=$is_peft \
    peft_config.r=$r \
    peft_config.lora_alpha=$lora_alpha \
    peft_config.target_modules=$target_modules \
    peft_config.lora_dropout=$lora_dropout \
    max_length=$max_length \
    max_new_tokens=$max_new_tokens \
    is_bf16=$is_bf16 \
    batch_size=$batch_size \
    eval_batch_size=$eval_batch_size \
    gradient_accumulation_steps=$gradient_accumulation_steps \
    epsilon=$epsilon \
    epsilon_high=$epsilon_high \
    num_generations=$num_generations \
    generation_config.temperature=$temperature \
    log_completions=$log_completions \
    num_completions_to_print=$num_completions_to_print \
    vllm_server_base_url=$vllm_server_base_url \
    vllm_server_timeout=$vllm_server_timeout \
    request_timeout=$request_timeout \
    max_inflight_tasks=$max_inflight_tasks \
    max_staleness=$max_staleness \
    queue_maxsize=$queue_maxsize \
    weight_sync_steps=$weight_sync_steps \
    async_runtime.enable=false \
    reward.is_answer_tag=$is_answer_tag \
    "${reward_weight_args[@]}" \
    lr=$lr \
    weight_decay=$weight_decay \
    warmup_ratio=$warmup_ratio \
    epoch=$epoch \
    step=$step \
    workers_ratio=$workers_ratio \
    use_all_workers=$use_all_workers