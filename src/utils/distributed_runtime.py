from typing import Dict, List, Optional, Any
import os
import socket
import sys

from omegaconf import DictConfig, ListConfig, OmegaConf

import torch
from torch import distributed as dist


def build_distributed_runtime_snapshot(
    config: DictConfig,
) -> Dict[str, Any]:
    observed_distributed = _build_observed_distributed_section()
    planned_distributed = _build_planned_distributed_section(
        config=config,
        observed_distributed=observed_distributed,
    )
    device_runtime = _build_device_runtime_section(
        config=config,
        observed_distributed=observed_distributed,
    )
    batch_runtime = _build_batch_runtime_section(
        config=config,
        planned_distributed=planned_distributed,
        observed_distributed=observed_distributed,
        device_runtime=device_runtime,
    )
    host_runtime = _build_host_runtime_section()

    return {
        "devices": _get_config_value(
            config=config,
            path="devices",
            default=None,
        ),
        "effective_batch_size": batch_runtime["planned_effective_train_batch_size"],
        "cuda_visible_devices": device_runtime["cuda_visible_devices"],
        "world_size": observed_distributed["world_size"],
        "rank_zero_host": host_runtime["hostname"],
        "distributed": {
            "planned": planned_distributed,
            "observed": observed_distributed,
        },
        "device": device_runtime,
        "batch": batch_runtime,
        "host": host_runtime,
    }


def _build_planned_distributed_section(
    config: DictConfig,
    observed_distributed: Dict[str, Any],
) -> Dict[str, Any]:
    distributed_enabled = bool(
        _get_config_value(
            config=config,
            path="distributed.enabled",
            default=False,
        )
    )
    num_machines = int(
        _get_config_value(
            config=config,
            path="distributed.num_machines",
            default=1,
        )
    )
    configured_num_processes_per_machine = _get_config_value(
        config=config,
        path="distributed.num_processes_per_machine",
        default=None,
    )
    num_processes_per_machine = _resolve_num_processes_per_machine(
        config=config,
        configured_num_processes_per_machine=configured_num_processes_per_machine,
        observed_distributed=observed_distributed,
    )

    return {
        "enabled": distributed_enabled,
        "num_machines": num_machines,
        "num_processes_per_machine": num_processes_per_machine,
        "configured_num_processes_per_machine": configured_num_processes_per_machine,
        "machine_rank": _get_config_value(
            config=config,
            path="distributed.machine_rank",
            default=0,
        ),
        "main_process_ip": _get_config_value(
            config=config,
            path="distributed.main_process_ip",
            default="127.0.0.1",
        ),
        "main_process_port": _get_config_value(
            config=config,
            path="distributed.main_process_port",
            default=29500,
        ),
        "world_size": num_machines * num_processes_per_machine,
        "validation_mode": _get_config_value(
            config=config,
            path="distributed.validation_mode",
            default="warn",
        ),
    }


def _build_observed_distributed_section() -> Dict[str, Any]:
    torch_distributed_available = dist.is_available()
    torch_distributed_initialized = (
        torch_distributed_available and dist.is_initialized()
    )
    return {
        "rank": _get_env_int(
            name="RANK",
            default=0,
        ),
        "local_rank": _get_env_int(
            name="LOCAL_RANK",
            default=0,
        ),
        "world_size": _get_env_int(
            name="WORLD_SIZE",
            default=1,
        ),
        "local_world_size": _get_env_int(
            name="LOCAL_WORLD_SIZE",
            default=1,
        ),
        "master_addr": os.environ.get("MASTER_ADDR"),
        "master_port": os.environ.get("MASTER_PORT"),
        "torchelastic_run_id": os.environ.get("TORCHELASTIC_RUN_ID"),
        "torch_distributed_available": torch_distributed_available,
        "torch_distributed_initialized": torch_distributed_initialized,
        "torch_distributed_backend": (
            dist.get_backend() if torch_distributed_initialized else None
        ),
        "torch_distributed_world_size": (
            dist.get_world_size() if torch_distributed_initialized else None
        ),
        "torch_distributed_rank": (
            dist.get_rank() if torch_distributed_initialized else None
        ),
    }


def _build_device_runtime_section(
    config: DictConfig,
    observed_distributed: Dict[str, Any],
) -> Dict[str, Any]:
    config_devices = _get_config_value(
        config=config,
        path="devices",
        default=None,
    )
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected_device_ids = _resolve_selected_device_ids(
        config_devices=config_devices,
        cuda_visible_devices=cuda_visible_devices,
    )
    torch_cuda_available = torch.cuda.is_available()
    torch_cuda_device_count = torch.cuda.device_count()
    current_device = (
        torch.cuda.current_device()
        if torch_cuda_available and torch_cuda_device_count > 0
        else None
    )

    return {
        "config_devices": config_devices,
        "cuda_visible_devices": cuda_visible_devices,
        "selected_device_ids": selected_device_ids,
        "selected_device_count": len(selected_device_ids),
        "torch_cuda_available": torch_cuda_available,
        "torch_cuda_device_count": torch_cuda_device_count,
        "torch_cuda_current_device": current_device,
        "torch_cuda_current_device_name": (
            torch.cuda.get_device_name(current_device)
            if current_device is not None
            else None
        ),
        "expected_local_device": observed_distributed["local_rank"],
    }


def _build_batch_runtime_section(
    config: DictConfig,
    planned_distributed: Dict[str, Any],
    observed_distributed: Dict[str, Any],
    device_runtime: Dict[str, Any],
) -> Dict[str, Any]:
    per_device_train_batch_size = int(config.batch_size)
    per_device_eval_batch_size = int(config.eval_batch_size)
    gradient_accumulation_steps = int(config.gradient_accumulation_steps)
    local_process_count = _resolve_local_process_count(
        planned_distributed=planned_distributed,
        observed_distributed=observed_distributed,
        device_runtime=device_runtime,
    )
    planned_world_size = int(planned_distributed["world_size"])
    observed_world_size = int(observed_distributed["world_size"])

    return {
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "local_process_count": local_process_count,
        "planned_world_size": planned_world_size,
        "observed_world_size": observed_world_size,
        "local_effective_train_batch_size": (
            per_device_train_batch_size
            * local_process_count
            * gradient_accumulation_steps
        ),
        "planned_effective_train_batch_size": (
            per_device_train_batch_size
            * planned_world_size
            * gradient_accumulation_steps
        ),
        "observed_effective_train_batch_size": (
            per_device_train_batch_size
            * observed_world_size
            * gradient_accumulation_steps
        ),
    }


def _build_host_runtime_section() -> Dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
    }


def _resolve_num_processes_per_machine(
    config: DictConfig,
    configured_num_processes_per_machine: Any,
    observed_distributed: Dict[str, Any],
) -> int:
    if configured_num_processes_per_machine is not None:
        return _get_device_count_from_value(value=configured_num_processes_per_machine)

    config_device_count = _get_device_count_from_value(
        value=_get_config_value(
            config=config,
            path="devices",
            default=None,
        )
    )
    if config_device_count is not None:
        return config_device_count

    return int(observed_distributed["local_world_size"])


def _resolve_selected_device_ids(
    config_devices: Any,
    cuda_visible_devices: Optional[str],
) -> List[str]:
    if cuda_visible_devices:
        return _split_device_ids(value=cuda_visible_devices)

    if isinstance(config_devices, int):
        return [str(device_idx) for device_idx in range(config_devices)]

    return _split_device_ids(value=config_devices)


def _resolve_local_process_count(
    planned_distributed: Dict[str, Any],
    observed_distributed: Dict[str, Any],
    device_runtime: Dict[str, Any],
) -> int:
    if int(observed_distributed["local_world_size"]) > 1:
        return int(observed_distributed["local_world_size"])

    if int(device_runtime["selected_device_count"]) > 0:
        return int(device_runtime["selected_device_count"])

    return int(planned_distributed["num_processes_per_machine"])


def _get_device_count_from_value(
    value: Any,
) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    device_ids = _split_device_ids(value=value)
    if len(device_ids) > 0:
        return len(device_ids)
    return None


def _split_device_ids(
    value: Any,
) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [
            device_id.strip() for device_id in value.split(",") if device_id.strip()
        ]
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(device_id) for device_id in value]
    return []


def _get_env_int(
    name: str,
    default: int,
) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _get_config_value(
    config: DictConfig,
    path: str,
    default: Any,
) -> Any:
    value = OmegaConf.select(
        config,
        path,
        default=default,
    )
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(
            value,
            resolve=True,
        )
    return value
