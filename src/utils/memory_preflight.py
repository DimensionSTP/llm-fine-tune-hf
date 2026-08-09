from typing import Dict, List, Union, Optional, Any
import os
import json
import subprocess
import sys
import time

from omegaconf import DictConfig, ListConfig, OmegaConf

import torch
from torch.utils.data import Dataset, Subset


def validate_memory_preflight_config(
    config: DictConfig,
) -> None:
    if config.memory_preflight.strategy != "strict_subprocess":
        raise ValueError("memory_preflight.strategy must be strict_subprocess.")
    if config.memory_preflight.batch_selection != "max_shape":
        raise ValueError("memory_preflight.batch_selection must be max_shape.")
    if int(config.memory_preflight.probe_steps) <= 0:
        raise ValueError("memory_preflight.probe_steps must be positive.")
    if config.memory_preflight.include_generation not in ["auto", True, False]:
        raise ValueError(
            "memory_preflight.include_generation must be auto, true, or false."
        )
    if not config.memory_preflight.enabled:
        return
    if not config.memory_preflight.fail_on_oom:
        raise ValueError(
            "memory_preflight.fail_on_oom=false is not supported. "
            "Disable memory_preflight instead of continuing after a failed probe."
        )
    if not config.memory_preflight.include_backward:
        raise ValueError("memory_preflight.include_backward=false is not supported.")
    if not config.memory_preflight.include_optimizer_step:
        raise ValueError(
            "memory_preflight.include_optimizer_step=false is not supported."
        )
    if config.memory_preflight.include_generation != "auto":
        raise ValueError("memory_preflight.include_generation must be auto in v1.")


def run_memory_preflight_if_needed(
    config: DictConfig,
    rank: int,
) -> None:
    validate_memory_preflight_config(config=config)
    if not config.memory_preflight.enabled:
        return
    if config.memory_preflight.is_probe:
        return
    _validate_supported_memory_preflight_method(config=config)

    runtime = _build_memory_preflight_runtime(config=config)
    if runtime["world_size"] > 1:
        _run_single_node_distributed_memory_preflight(
            config=config,
            rank=rank,
            runtime=runtime,
        )
        return
    if rank != 0:
        return
    _run_memory_preflight_probe(
        config=config,
        runtime=runtime,
    )


def apply_memory_preflight_dataset(
    config: DictConfig,
    train_dataset: Union[Dataset, Any],
) -> Union[Dataset, Any]:
    validate_memory_preflight_config(config=config)
    if not config.memory_preflight.is_probe:
        return train_dataset
    _validate_supported_memory_preflight_method(config=config)
    selected_indices = _load_memory_preflight_selected_indices(
        path=config.memory_preflight.selected_indices_path,
    )
    if hasattr(train_dataset, "select"):
        return train_dataset.select(selected_indices)
    return Subset(
        train_dataset,
        selected_indices,
    )


def write_memory_preflight_selection(
    config: DictConfig,
    train_dataset: Union[Dataset, Any],
) -> None:
    if not config.memory_preflight.is_probe:
        return
    selected_indices_path = config.memory_preflight.selected_indices_path
    if selected_indices_path is None:
        raise ValueError(
            "memory_preflight.selected_indices_path is required in probe mode."
        )
    if _get_memory_preflight_world_size() > 1 and _get_memory_preflight_rank() != 0:
        _wait_for_memory_preflight_file(
            config=config,
            path=selected_indices_path,
        )
        return
    selection = _select_memory_preflight_indices(
        config=config,
        train_dataset=train_dataset,
    )
    _write_memory_preflight_json(
        path=selected_indices_path,
        payload=selection,
    )


def build_memory_preflight_metadata(
    config: DictConfig,
) -> Dict[str, Any]:
    metadata = OmegaConf.to_container(
        config.memory_preflight,
        resolve=True,
    )
    if not isinstance(metadata, dict):
        raise ValueError("memory_preflight metadata must be a dictionary.")
    return metadata


def _run_single_node_distributed_memory_preflight(
    config: DictConfig,
    rank: int,
    runtime: Dict[str, Any],
) -> None:
    if (
        runtime["local_world_size"] != runtime["world_size"]
        or int(config.distributed.num_machines) != 1
    ):
        raise ValueError(
            "memory_preflight strict_subprocess currently supports direct "
            "single-node distributed launch only. Multi-node, vLLM server, and "
            "async topologies require lifecycle-aware preflight."
        )

    coordination_path = str(runtime["coordination_path"])
    if rank == 0:
        result_payload = _run_memory_preflight_probe(
            config=config,
            runtime=runtime,
        )
    else:
        coordination = _wait_for_memory_preflight_coordination(
            config=config,
            coordination_path=coordination_path,
            runtime=runtime,
        )
        result_payload = _wait_for_memory_preflight_result(
            config=config,
            result_path=str(coordination["result_path"]),
        )

    if result_payload["exit_code"] == 0:
        return
    raise RuntimeError(
        "Memory preflight failed before training. "
        f"probe_id={result_payload['probe_id']}, "
        f"exit_code={result_payload['exit_code']}, "
        f"probe_dir={result_payload['probe_dir']}"
    )


def _run_memory_preflight_probe(
    config: DictConfig,
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    probe_id = _build_memory_preflight_probe_id(runtime=runtime)
    probe_dir = _build_memory_preflight_probe_dir(
        config=config,
        probe_id=probe_id,
    )
    os.makedirs(
        probe_dir,
        exist_ok=True,
    )
    selected_indices_path = os.path.join(
        probe_dir,
        "selected_indices.json",
    )
    command = _build_memory_preflight_command(
        config=config,
        probe_id=probe_id,
        probe_dir=probe_dir,
        selected_indices_path=selected_indices_path,
        runtime=runtime,
    )
    result_path = os.path.join(
        probe_dir,
        "result.json",
    )
    if runtime["world_size"] > 1:
        _write_memory_preflight_json(
            path=str(runtime["coordination_path"]),
            payload={
                "probe_id": probe_id,
                "probe_dir": probe_dir,
                "result_path": result_path,
                "selected_indices_path": selected_indices_path,
                "created_at": time.time(),
            },
        )
    _write_memory_preflight_json(
        path=os.path.join(
            probe_dir,
            "command.json",
        ),
        payload={
            "probe_id": probe_id,
            "command": command,
            "selected_indices_path": selected_indices_path,
        },
    )
    result = subprocess.run(
        command,
        cwd=str(config.work_dir),
        env=_build_memory_preflight_environment(
            config=config,
            runtime=runtime,
        ),
        text=True,
    )
    result_payload = {
        "probe_id": probe_id,
        "exit_code": int(result.returncode),
        "success": result.returncode == 0,
        "probe_dir": probe_dir,
        "selected_indices_path": selected_indices_path,
    }
    _write_memory_preflight_json(
        path=result_path,
        payload=result_payload,
    )
    if result.returncode == 0:
        return result_payload
    raise RuntimeError(
        "Memory preflight failed before training. "
        f"probe_id={probe_id}, exit_code={result.returncode}, probe_dir={probe_dir}"
    )


def _validate_supported_memory_preflight_method(
    config: DictConfig,
) -> None:
    supported_methods = [
        "sft",
        "dpo",
        "kto",
        "gkd",
        "gold",
        "grpo",
        "sdpo",
        "a2po",
    ]
    if config.fine_tune_method not in supported_methods:
        raise ValueError(
            "memory_preflight strict_subprocess currently supports "
            "sft, dpo, kto, gkd, gold, grpo, sdpo, and a2po."
        )
    if (
        config.fine_tune_method in {"gold", "grpo", "sdpo"}
        and config.use_vllm
        and config.vllm_mode != "colocate"
    ):
        raise ValueError(
            "memory_preflight strict_subprocess currently supports vLLM colocate "
            "mode only. vLLM server and async topologies require lifecycle-aware "
            "preflight."
        )


def _select_memory_preflight_indices(
    config: DictConfig,
    train_dataset: Dataset,
) -> Dict[str, Any]:
    validate_memory_preflight_config(config=config)
    _validate_supported_memory_preflight_method(config=config)

    required_count = int(config.batch_size) * int(config.gradient_accumulation_steps)
    if required_count <= 0:
        raise ValueError("memory_preflight requires a positive probe sample count.")
    if len(train_dataset) == 0:
        raise ValueError("memory_preflight cannot select from an empty train dataset.")

    scored_samples = [
        _score_memory_preflight_sample(
            sample=train_dataset[index],
            index=index,
        )
        for index in range(len(train_dataset))
    ]
    selected = sorted(
        scored_samples,
        key=lambda item: (item["score"], item["sequence_length"], item["index"]),
        reverse=True,
    )[:required_count]
    indices = [int(item["index"]) for item in selected]
    return {
        "indices": indices,
        "required_count": required_count,
        "selected_count": len(indices),
        "dataset_length": len(train_dataset),
        "generation": _build_memory_preflight_generation_metadata(config=config),
        "samples": selected,
    }


def _build_memory_preflight_generation_metadata(
    config: DictConfig,
) -> Dict[str, Any]:
    method = str(config.fine_tune_method)
    metadata = {
        "fine_tune_method": method,
    }
    if method in {"grpo", "sdpo", "async_grpo"}:
        metadata.update(
            {
                "num_generations": int(config.num_generations),
                "max_completion_length": int(config.max_new_tokens),
                "steps_per_generation": config.steps_per_generation,
                "use_vllm": bool(config.use_vllm),
            }
        )
        if config.use_vllm:
            metadata.update(
                {
                    "vllm_mode": str(config.vllm_mode),
                    "vllm_tensor_parallel_size": int(config.vllm_tensor_parallel_size),
                    "gpu_memory_utilization": float(config.gpu_memory_utilization),
                }
            )
    if method == "a2po":
        metadata.update(
            {
                "num_value_samples": int(config.num_value_samples),
                "max_completion_length": int(config.max_new_tokens),
            }
        )
    return metadata


def _score_memory_preflight_sample(
    sample: Dict[str, Any],
    index: int,
) -> Dict[str, Any]:
    tensor_shapes = _build_memory_preflight_tensor_shapes(sample=sample)
    tensor_numel = sum(item["numel"] for item in tensor_shapes.values())
    sequence_length = _resolve_memory_preflight_sequence_length(sample=sample)
    payload_size = _score_memory_preflight_payload(value=sample)
    score = tensor_numel if tensor_numel > 0 else payload_size
    return {
        "index": index,
        "score": int(score),
        "sequence_length": int(sequence_length),
        "payload_size": int(payload_size),
        "tensor_shapes": tensor_shapes,
    }


def _build_memory_preflight_tensor_shapes(
    sample: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return {
        key: {
            "shape": list(value.shape),
            "numel": int(value.numel()),
            "dtype": str(value.dtype),
        }
        for key, value in sample.items()
        if isinstance(value, torch.Tensor)
    }


def _resolve_memory_preflight_sequence_length(
    sample: Dict[str, Any],
) -> int:
    if "input_ids" in sample:
        input_ids = sample["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("memory_preflight sample input_ids must be a tensor.")
        return int(input_ids.size(0))
    return _score_memory_preflight_payload(value=sample)


def _score_memory_preflight_payload(
    value: Any,
) -> int:
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    if isinstance(value, str):
        return len(value)
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, dict):
        return sum(
            _score_memory_preflight_payload(value=item_value)
            for item_value in value.values()
        )
    if isinstance(value, list):
        return sum(_score_memory_preflight_payload(value=item) for item in value)
    if isinstance(value, tuple):
        return sum(_score_memory_preflight_payload(value=item) for item in value)
    return len(str(value))


def _build_memory_preflight_command(
    config: DictConfig,
    probe_id: str,
    probe_dir: str,
    selected_indices_path: str,
    runtime: Dict[str, Any],
) -> List[str]:
    output_base_dir = os.path.join(
        probe_dir,
        "checkpoints",
    )
    overrides = [
        "memory_preflight.is_probe=true",
        f"memory_preflight.probe_id={probe_id}",
        "memory_preflight.selected_indices_path="
        f"{_escape_memory_preflight_override_value(value=selected_indices_path)}",
        f"output_base_dir={_escape_memory_preflight_override_value(value=output_base_dir)}",
        "run_id=null",
        "output_dir=null",
        "use_validation=false",
        "val_dataset_file_path=null",
        "val_dataset_file_paths=null",
        "val_dataset_files=null",
        "allow_val_dataset_file_name_mismatch=false",
        "eval_strategy=no",
        "save_strategy=no",
        "save_total_limit=1",
        "tracking.report_to=none",
        f"++training_arguments.max_steps={int(config.memory_preflight.probe_steps)}",
    ]
    if runtime["world_size"] <= 1:
        return [
            sys.executable,
            *_filter_memory_preflight_argv(argv=sys.argv),
            *overrides,
        ]
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_machines=1",
        f"--num_processes={int(runtime['world_size'])}",
        "--machine_rank=0",
        "--main_process_ip=127.0.0.1",
        f"--main_process_port={int(runtime['probe_port'])}",
        *_build_memory_preflight_gpu_id_args(config=config),
        *_filter_memory_preflight_argv(argv=sys.argv),
        "distributed.enabled=true",
        "distributed.num_machines=1",
        f"distributed.num_processes_per_machine={int(runtime['world_size'])}",
        "distributed.machine_rank=0",
        "distributed.main_process_ip=127.0.0.1",
        f"distributed.main_process_port={int(runtime['probe_port'])}",
        *overrides,
    ]


def _build_memory_preflight_probe_id(
    runtime: Dict[str, Any],
) -> str:
    if runtime["world_size"] > 1:
        return f"probe-distributed-{os.getpid()}"
    return f"probe-{os.getpid()}"


def _filter_memory_preflight_argv(
    argv: List[str],
) -> List[str]:
    removed_prefixes = [
        "memory_preflight.is_probe=",
        "memory_preflight.probe_id=",
        "memory_preflight.selected_indices_path=",
        "output_base_dir=",
        "run_id=",
        "output_dir=",
        "use_validation=",
        "val_dataset_file_path=",
        "val_dataset_file_paths=",
        "val_dataset_files=",
        "allow_val_dataset_file_name_mismatch=",
        "eval_strategy=",
        "save_strategy=",
        "save_total_limit=",
        "tracking.report_to=",
        "training_arguments.max_steps=",
        "+training_arguments.max_steps=",
        "++training_arguments.max_steps=",
    ]
    return [
        item
        for item in argv
        if not any(item.startswith(prefix) for prefix in removed_prefixes)
    ]


def _escape_memory_preflight_override_value(
    value: str,
) -> str:
    return (
        str(value)
        .replace(
            "\\",
            "\\\\",
        )
        .replace(
            "=",
            "\\=",
        )
    )


def _build_memory_preflight_probe_dir(
    config: DictConfig,
    probe_id: str,
) -> str:
    return os.path.join(
        str(config.output_base_dir),
        ".memory_preflight",
        probe_id,
    )


def _build_memory_preflight_runtime(
    config: DictConfig,
) -> Dict[str, Any]:
    world_size = _get_memory_preflight_world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    return {
        "world_size": world_size,
        "local_world_size": local_world_size,
        "master_addr": os.environ.get("MASTER_ADDR", "127.0.0.1"),
        "master_port": os.environ.get("MASTER_PORT", "0"),
        "probe_port": _build_memory_preflight_probe_port(
            config=config,
            world_size=world_size,
        ),
        "started_at": time.time(),
        "coordination_path": _build_memory_preflight_coordination_path(
            config=config,
            runtime={
                "world_size": world_size,
                "master_addr": os.environ.get("MASTER_ADDR", "127.0.0.1"),
                "master_port": os.environ.get("MASTER_PORT", "0"),
            },
        ),
    }


def _build_memory_preflight_coordination_path(
    config: DictConfig,
    runtime: Dict[str, Any],
) -> str:
    master_addr = _sanitize_memory_preflight_path_token(
        value=str(runtime["master_addr"]),
    )
    master_port = _sanitize_memory_preflight_path_token(
        value=str(runtime["master_port"]),
    )
    return os.path.join(
        str(config.output_base_dir),
        ".memory_preflight",
        f"distributed_{master_addr}_{master_port}_{int(runtime['world_size'])}.json",
    )


def _build_memory_preflight_environment(
    config: DictConfig,
    runtime: Dict[str, Any],
) -> Dict[str, str]:
    environment = dict(os.environ)
    environment["WANDB_MODE"] = "offline"
    if runtime["world_size"] > 1:
        for key in _get_memory_preflight_distributed_env_keys():
            environment.pop(
                key,
                None,
            )
    cuda_visible_devices = _resolve_memory_preflight_cuda_visible_devices(
        config=config,
    )
    if cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return environment


def _resolve_memory_preflight_cuda_visible_devices(
    config: DictConfig,
) -> Optional[str]:
    devices = config.devices
    if devices is None:
        return None
    if isinstance(devices, int):
        if torch.cuda.device_count() <= 0 and os.environ.get("CUDA_VISIBLE_DEVICES"):
            return os.environ["CUDA_VISIBLE_DEVICES"]
        num_gpus = min(
            devices,
            torch.cuda.device_count(),
        )
        return ",".join(map(str, range(num_gpus)))
    if isinstance(devices, str):
        return devices
    if isinstance(devices, (list, ListConfig)):
        return ",".join(map(str, devices))
    raise ValueError("memory_preflight devices must be int, str, list, or null.")


def _build_memory_preflight_gpu_id_args(
    config: DictConfig,
) -> List[str]:
    cuda_visible_devices = _resolve_memory_preflight_cuda_visible_devices(
        config=config,
    )
    if cuda_visible_devices is None:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is None or cuda_visible_devices == "":
        return []
    return [
        f"--gpu_ids={cuda_visible_devices}",
    ]


def _get_memory_preflight_distributed_env_keys() -> List[str]:
    return [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
    ]


def _build_memory_preflight_probe_port(
    config: DictConfig,
    world_size: int,
) -> Optional[int]:
    if world_size <= 1:
        return None
    parent_port = int(config.distributed.main_process_port)
    probe_port = 20000 + ((parent_port + os.getpid()) % 40000)
    if probe_port == parent_port:
        return probe_port + 1
    return probe_port


def _wait_for_memory_preflight_coordination(
    config: DictConfig,
    coordination_path: str,
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    deadline_at = time.monotonic() + float(
        config.run_metadata.allocation_timeout_seconds
    )
    poll_interval_seconds = float(config.run_metadata.allocation_poll_interval_seconds)
    freshness_grace_seconds = float(
        config.run_metadata.allocation_freshness_grace_seconds
    )
    while time.monotonic() < deadline_at:
        if os.path.isfile(coordination_path):
            with open(coordination_path, encoding="utf-8") as file:
                payload = json.load(file)
            created_at = payload["created_at"]
            if created_at >= runtime["started_at"] - freshness_grace_seconds:
                return payload
        time.sleep(poll_interval_seconds)
    raise TimeoutError(
        f"Timed out waiting for memory preflight coordination: {coordination_path}"
    )


def _wait_for_memory_preflight_result(
    config: DictConfig,
    result_path: str,
) -> Dict[str, Any]:
    _wait_for_memory_preflight_file(
        config=config,
        path=result_path,
    )
    with open(result_path, encoding="utf-8") as file:
        return json.load(file)


def _wait_for_memory_preflight_file(
    config: DictConfig,
    path: str,
) -> None:
    deadline_at = time.monotonic() + float(
        config.run_metadata.allocation_timeout_seconds
    )
    poll_interval_seconds = float(config.run_metadata.allocation_poll_interval_seconds)
    while time.monotonic() < deadline_at:
        if os.path.isfile(path):
            return
        time.sleep(poll_interval_seconds)
    raise TimeoutError(f"Timed out waiting for memory preflight file: {path}")


def _get_memory_preflight_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _get_memory_preflight_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _sanitize_memory_preflight_path_token(
    value: str,
) -> str:
    return (
        value.replace(
            "/",
            "_",
        )
        .replace(
            ":",
            "_",
        )
        .replace(
            ".",
            "_",
        )
    )


def _load_memory_preflight_selected_indices(
    path: str,
) -> List[int]:
    if path is None:
        raise ValueError("memory_preflight.selected_indices_path is required.")
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    indices = payload["indices"]
    if not isinstance(indices, list) or len(indices) == 0:
        raise ValueError("memory_preflight selected indices must be a non-empty list.")
    return [int(index) for index in indices]


def _write_memory_preflight_json(
    path: str,
    payload: Dict[str, Any],
) -> None:
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            _make_memory_preflight_jsonable(value=payload),
            file,
            ensure_ascii=False,
            indent=2,
        )


def _make_memory_preflight_jsonable(
    value: Any,
) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(
            value,
            resolve=True,
        )
    if isinstance(value, dict):
        return {
            str(item_key): _make_memory_preflight_jsonable(value=item_value)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_make_memory_preflight_jsonable(value=item) for item in value]
    if isinstance(value, tuple):
        return [_make_memory_preflight_jsonable(value=item) for item in value]
    return value
