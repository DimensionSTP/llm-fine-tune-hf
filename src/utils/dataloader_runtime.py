from typing import Dict, Optional, Any
import os

from omegaconf import DictConfig


def resolve_dataloader_runtime(
    config: DictConfig,
    distributed_runtime_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    validate_dataloader_runtime_config(config=config)

    workload = _resolve_dataloader_workload(config=config)
    cpu_count = os.cpu_count()
    local_world_size = _resolve_local_world_size(
        distributed_runtime_snapshot=distributed_runtime_snapshot,
    )
    num_workers_per_process = _resolve_num_workers_per_process(
        config=config,
        cpu_count=cpu_count,
        local_world_size=local_world_size,
    )
    persistent_workers = _resolve_persistent_workers(
        config=config,
        num_workers_per_process=num_workers_per_process,
    )
    prefetch_factor = _resolve_prefetch_factor(
        config=config,
        workload=workload,
        num_workers_per_process=num_workers_per_process,
    )

    _validate_resolved_dataloader_runtime(
        num_workers_per_process=num_workers_per_process,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    total_workers_per_node = num_workers_per_process * local_world_size
    prefetch_slots_per_process = _resolve_prefetch_slots_per_process(
        num_workers_per_process=num_workers_per_process,
        prefetch_factor=prefetch_factor,
    )

    return {
        "workload": workload,
        "cpu_count": cpu_count,
        "num_workers_per_process": num_workers_per_process,
        "total_workers_per_node": total_workers_per_node,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "prefetch_slots_per_process": prefetch_slots_per_process,
        "prefetch_slots_per_node": prefetch_slots_per_process * local_world_size,
        "pin_memory": bool(config.dataloader_runtime.pin_memory),
    }


def validate_dataloader_runtime_config(
    config: DictConfig,
) -> None:
    mode = str(config.dataloader_runtime.mode)
    if mode not in ("auto", "manual"):
        raise ValueError("dataloader_runtime.mode must be auto or manual.")
    if mode == "manual" and config.dataloader_runtime.num_workers_per_process is None:
        raise ValueError(
            "dataloader_runtime.num_workers_per_process is required when mode is manual."
        )

    workload = str(config.dataloader_runtime.workload)
    if workload not in ("auto", "text", "vlm", "vlm_decode", "vlm_aug"):
        raise ValueError(
            "dataloader_runtime.workload must be auto, text, vlm, vlm_decode, or vlm_aug."
        )

    if int(config.dataloader_runtime.max_workers_per_process) < 0:
        raise ValueError(
            "dataloader_runtime.max_workers_per_process must be greater than or equal to 0."
        )

    if config.dataloader_runtime.num_workers_per_process is not None:
        if int(config.dataloader_runtime.num_workers_per_process) < 0:
            raise ValueError(
                "dataloader_runtime.num_workers_per_process must be greater than or equal to 0."
            )

    if config.dataloader_runtime.prefetch_factor is not None:
        if int(config.dataloader_runtime.prefetch_factor) <= 0:
            raise ValueError("dataloader_runtime.prefetch_factor must be positive.")


def _resolve_dataloader_workload(
    config: DictConfig,
) -> str:
    configured_workload = str(config.dataloader_runtime.workload)
    if configured_workload != "auto":
        return configured_workload

    if config.modality == "text":
        return "text"

    if "image_augmentation" in config and config.image_augmentation.enabled:
        return "vlm_aug"

    if "decode_image_paths" in config and config.decode_image_paths:
        return "vlm_decode"

    return "vlm"


def _resolve_local_world_size(
    distributed_runtime_snapshot: Dict[str, Any],
) -> int:
    observed_local_world_size = int(
        distributed_runtime_snapshot["distributed"]["observed"]["local_world_size"]
    )
    if observed_local_world_size > 0:
        return observed_local_world_size

    planned_local_world_size = int(
        distributed_runtime_snapshot["distributed"]["planned"][
            "num_processes_per_machine"
        ]
    )
    if planned_local_world_size <= 0:
        raise ValueError("dataloader runtime requires a positive local_world_size.")
    return planned_local_world_size


def _resolve_num_workers_per_process(
    config: DictConfig,
    cpu_count: Optional[int],
    local_world_size: int,
) -> int:
    if "dataset_streaming" in config and config.dataset_streaming.enabled:
        return 0

    configured_workers = config.dataloader_runtime.num_workers_per_process
    if configured_workers is not None:
        return int(configured_workers)

    if local_world_size <= 0:
        raise ValueError("dataloader runtime requires a positive local_world_size.")

    if cpu_count is None:
        return 0
    if cpu_count < local_world_size:
        return 0

    cpu_budget_per_process = cpu_count // local_world_size
    return min(
        cpu_budget_per_process,
        int(config.dataloader_runtime.max_workers_per_process),
    )


def _resolve_persistent_workers(
    config: DictConfig,
    num_workers_per_process: int,
) -> bool:
    configured_persistent_workers = config.dataloader_runtime.persistent_workers
    if configured_persistent_workers is not None:
        return bool(configured_persistent_workers)
    return num_workers_per_process > 0


def _resolve_prefetch_factor(
    config: DictConfig,
    workload: str,
    num_workers_per_process: int,
) -> Optional[int]:
    configured_prefetch_factor = config.dataloader_runtime.prefetch_factor
    if configured_prefetch_factor is not None:
        return int(configured_prefetch_factor)
    if num_workers_per_process == 0:
        return None
    return 2


def _resolve_prefetch_slots_per_process(
    num_workers_per_process: int,
    prefetch_factor: Optional[int],
) -> int:
    if num_workers_per_process == 0:
        return 0
    if prefetch_factor is None:
        return 0
    return num_workers_per_process * prefetch_factor


def _validate_resolved_dataloader_runtime(
    num_workers_per_process: int,
    persistent_workers: bool,
    prefetch_factor: Optional[int],
) -> None:
    if num_workers_per_process > 0:
        return
    if persistent_workers:
        raise ValueError(
            "dataloader_runtime.persistent_workers must be false when num_workers_per_process is 0."
        )
    if prefetch_factor is not None:
        raise ValueError(
            "dataloader_runtime.prefetch_factor must be null when num_workers_per_process is 0."
        )
