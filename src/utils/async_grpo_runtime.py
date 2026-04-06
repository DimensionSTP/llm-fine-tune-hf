from typing import Dict, List, Tuple, Union, Optional, Any

import json
import os
import shutil
import subprocess
import time
import urllib.request

from omegaconf import DictConfig


def resolve_async_half_gpu_partition(
) -> Dict[str, Union[str, int]]:
    visible_devices = os.environ.get(
        "CUDA_VISIBLE_DEVICES",
        "",
    ).strip()
    if visible_devices:
        visible_gpu_list = [
            gpu_id.strip()
            for gpu_id in visible_devices.split(",")
            if gpu_id.strip()
        ]
    elif shutil.which("nvidia-smi") is not None:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
        gpu_lines = result.stdout.splitlines()
        visible_gpu_list = [
            str(gpu_idx)
            for gpu_idx, gpu_line in enumerate(gpu_lines)
            if gpu_line.startswith("GPU ")
        ]
    else:
        raise ValueError(
            "async_grpo requires CUDA_VISIBLE_DEVICES or nvidia-smi to detect GPUs."
        )

    visible_gpu_count = len(visible_gpu_list)
    if visible_gpu_count < 2:
        raise ValueError(
            "async_grpo requires at least 2 visible GPUs for half-half partition."
        )
    if visible_gpu_count % 2 != 0:
        raise ValueError(
            "async_grpo half-half partition requires an even number of visible GPUs."
        )

    split_idx = visible_gpu_count // 2
    vllm_gpu_list = visible_gpu_list[:split_idx]
    trainer_gpu_list = visible_gpu_list[split_idx:]
    return {
        "visible_gpu_count": visible_gpu_count,
        "vllm_gpu_ids": ",".join(vllm_gpu_list),
        "trainer_gpu_ids": ",".join(trainer_gpu_list),
        "vllm_gpu_count": len(vllm_gpu_list),
        "trainer_gpu_count": len(trainer_gpu_list),
    }


def wait_vllm_server_ready(
    base_url: str,
    max_wait_sec: int,
) -> bool:
    target_url = f"{base_url.rstrip('/')}/v1/models"
    for _ in range(int(max_wait_sec)):
        try:
            with urllib.request.urlopen(
                target_url,
                timeout=1.0,
            ) as response:
                if response.status != 200:
                    time.sleep(1)
                    continue
                payload = json.loads(response.read().decode("utf-8"))
                if "data" in payload:
                    return True
        except Exception:
            time.sleep(1)
    return False


def start_async_vllm_server(
    config: DictConfig,
    *,
    vllm_gpu_ids: str,
) -> Tuple[subprocess.Popen, Any]:
    vllm_server_cfg = config.async_runtime.vllm_server

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = vllm_gpu_ids
    env.setdefault(
        "VLLM_SERVER_DEV_MODE",
        "1" if vllm_server_cfg.dev_mode else "0",
    )

    log_path = vllm_server_cfg.log_path
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(
            log_dir,
            exist_ok=True,
        )
    log_handle = open(
        log_path,
        "w",
        encoding="utf-8",
    )
    command = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        vllm_server_cfg.model,
        "--host",
        vllm_server_cfg.host,
        "--port",
        str(vllm_server_cfg.port),
        "--tensor-parallel-size",
        str(vllm_server_cfg.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(vllm_server_cfg.gpu_memory_utilization),
        "--weight-transfer-config",
        '{"backend":"nccl"}',
    ]
    process = subprocess.Popen(
        command,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return process, log_handle


def stop_async_vllm_server(
    *,
    process: Optional[subprocess.Popen],
    log_handle: Optional[Any],
) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(
                timeout=10,
            )

    if log_handle is not None and not log_handle.closed:
        log_handle.close()


def resolve_async_runtime_state(
    config: DictConfig,
    rank: int,
) -> Dict[str, Any]:
    enabled = (
        config.fine_tune_method == "async_grpo"
        and "async_runtime" in config
        and config.async_runtime.enable
    )
    state = {
        "enabled": enabled,
        "rank_is_inference": False,
        "server_managed_by_rank1": False,
        "stop_signal_path": None,
        "gpu_partition": None,
    }
    if not enabled:
        return state

    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )
    if world_size > 1:
        if world_size != 2:
            raise ValueError(
                "async_runtime.enable=true with distributed launch supports world_size=2 only "
                "(rank0=trainer, rank1=vllm server)."
            )

        config.vllm_server_base_url = (
            f"http://{config.async_runtime.vllm_server.host}:"
            f"{config.async_runtime.vllm_server.port}"
        )
        state["stop_signal_path"] = (
            f"/tmp/async_grpo_vllm_stop_{os.environ.get('MASTER_PORT', '29500')}.signal"
        )

        local_rank = int(
            os.environ.get(
                "LOCAL_RANK",
                str(rank),
            )
        )
        if rank == 1:
            state["rank_is_inference"] = True
            os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
            config.devices = 1
            return state

        if rank != 0:
            raise ValueError(
                f"Invalid rank for async_grpo world_size=2 split mode: rank={rank}"
            )

        state["server_managed_by_rank1"] = True
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        config.devices = 1
        config.async_runtime.vllm_server.auto_start = False
        return state

    gpu_partition = resolve_async_half_gpu_partition()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_partition["trainer_gpu_ids"]
    config.devices = gpu_partition["trainer_gpu_count"]
    config.vllm_server_base_url = (
        f"http://{config.async_runtime.vllm_server.host}:"
        f"{config.async_runtime.vllm_server.port}"
    )
    state["gpu_partition"] = gpu_partition
    return state


def run_async_inference_server(
    config: DictConfig,
    runtime_state: Dict[str, Any],
) -> bool:
    if not runtime_state["rank_is_inference"]:
        return False

    stop_signal_path = runtime_state["stop_signal_path"]
    if stop_signal_path is not None and os.path.exists(stop_signal_path):
        os.remove(stop_signal_path)

    process, log_handle = start_async_vllm_server(
        config=config,
        vllm_gpu_ids=os.environ["CUDA_VISIBLE_DEVICES"],
    )
    try:
        if not wait_vllm_server_ready(
            config.vllm_server_base_url,
            config.async_runtime.vllm_server.ready_timeout,
        ):
            raise RuntimeError(
                f"vLLM server did not become ready: {config.vllm_server_base_url}"
            )

        while True:
            if stop_signal_path is not None and os.path.exists(stop_signal_path):
                break
            if process.poll() is not None:
                raise RuntimeError("vLLM server process exited unexpectedly.")
            time.sleep(1)
    finally:
        stop_async_vllm_server(
            process=process,
            log_handle=log_handle,
        )
        if stop_signal_path is not None and os.path.exists(stop_signal_path):
            os.remove(stop_signal_path)
    return True


def start_async_training_runtime(
    config: DictConfig,
    runtime_state: Dict[str, Any],
) -> Tuple[Optional[Any], Optional[Any]]:
    if not runtime_state["enabled"]:
        return None, None

    if runtime_state["server_managed_by_rank1"]:
        if not wait_vllm_server_ready(
            config.vllm_server_base_url,
            config.async_runtime.vllm_server.ready_timeout,
        ):
            raise RuntimeError(
                f"vLLM server did not become ready: {config.vllm_server_base_url}"
            )
        return None, None

    if not config.async_runtime.vllm_server.auto_start:
        return None, None

    if wait_vllm_server_ready(config.vllm_server_base_url, 1):
        raise ValueError(
            f"Existing vLLM server already running at {config.vllm_server_base_url}"
        )

    process, log_handle = start_async_vllm_server(
        config=config,
        vllm_gpu_ids=runtime_state["gpu_partition"]["vllm_gpu_ids"],
    )
    if not wait_vllm_server_ready(
        config.vllm_server_base_url,
        config.async_runtime.vllm_server.ready_timeout,
    ):
        stop_async_vllm_server(
            process=process,
            log_handle=log_handle,
        )
        raise RuntimeError(f"vLLM server did not become ready: {config.vllm_server_base_url}")
    return process, log_handle


def stop_async_training_runtime(
    config: DictConfig,
    runtime_state: Dict[str, Any],
    process: Optional[Any],
    log_handle: Optional[Any],
) -> None:
    if runtime_state["server_managed_by_rank1"] and runtime_state["stop_signal_path"] is not None:
        with open(
            runtime_state["stop_signal_path"],
            "w",
            encoding="utf-8",
        ):
            pass

    if runtime_state["enabled"] and config.async_runtime.vllm_server.auto_stop:
        stop_async_vllm_server(
            process=process,
            log_handle=log_handle,
        )
