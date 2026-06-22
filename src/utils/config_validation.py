from typing import Dict, List, Optional, Any
import os

from omegaconf import DictConfig, ListConfig

from .distributed_runtime import build_distributed_runtime_snapshot
from .dataloader_runtime import validate_dataloader_runtime_config
from .model_loading import ModelLoadPlanner


def validate_training_arguments_config(
    config: DictConfig,
) -> None:
    _validate_dataset_input_config(config=config)

    if config.fine_tune_method == "async_grpo" and config.strategy == "deepspeed":
        raise ValueError(
            "async_grpo does not support strategy=deepspeed yet. "
            "Use strategy=none for the validated trainer/vLLM split path."
        )

    ModelLoadPlanner(
        config=config,
        torch_dtype="auto",
    ).validate()
    validate_dataloader_runtime_config(config=config)

    if config.fine_tune_method != "sft":
        return
    if (
        config.sft_loss_type == "chunked_nll"
        and config.training_arguments.use_liger_kernel
    ):
        raise ValueError(
            "SFT loss_type='chunked_nll' is not compatible with "
            "use_liger_kernel=True."
        )


def validate_train_artifact_config(
    config: DictConfig,
) -> None:
    if config.mode != "train":
        return

    if config.run_id is None:
        raise ValueError("run_id must be allocated before train starts.")

    if config.output_dir is None:
        raise ValueError("output_dir must be allocated before train starts.")

    output_dir = str(config.output_dir)
    if not os.path.exists(output_dir):
        raise ValueError(f"output_dir does not exist: {output_dir}")

    if not config.resume_training:
        return

    if config.resume_from_checkpoint is None:
        raise ValueError(
            "resume_from_checkpoint is required when resume_training is true."
        )

    resume_from_checkpoint = str(config.resume_from_checkpoint)
    if not os.path.exists(resume_from_checkpoint):
        raise ValueError(
            f"resume_from_checkpoint does not exist: {resume_from_checkpoint}"
        )


def validate_distributed_runtime_config(
    config: DictConfig,
    runtime_snapshot: Optional[Dict[str, Any]],
) -> None:
    if config.mode != "train":
        return

    snapshot = runtime_snapshot
    if snapshot is None:
        snapshot = build_distributed_runtime_snapshot(config=config)

    planned_distributed = snapshot["distributed"]["planned"]
    if not planned_distributed["enabled"]:
        return

    validation_mode = str(planned_distributed["validation_mode"])
    if validation_mode not in {"warn", "error"}:
        raise ValueError(
            "distributed.validation_mode must be either 'warn' or 'error'."
        )

    messages = _build_distributed_validation_messages(runtime_snapshot=snapshot)
    if len(messages) == 0:
        return

    if validation_mode == "error":
        raise ValueError("\n".join(messages))

    for message in messages:
        print(f"[distributed][warn] {message}")


def _build_distributed_validation_messages(
    runtime_snapshot: Dict[str, Any],
) -> List[str]:
    planned_distributed = runtime_snapshot["distributed"]["planned"]
    observed_distributed = runtime_snapshot["distributed"]["observed"]
    device_runtime = runtime_snapshot["device"]
    messages = []

    if planned_distributed["world_size"] != observed_distributed["world_size"]:
        messages.append(
            "planned world_size="
            f"{planned_distributed['world_size']} but observed WORLD_SIZE="
            f"{observed_distributed['world_size']}."
        )

    if (
        observed_distributed["local_world_size"] > 1
        and planned_distributed["num_processes_per_machine"]
        != observed_distributed["local_world_size"]
    ):
        messages.append(
            "planned num_processes_per_machine="
            f"{planned_distributed['num_processes_per_machine']} but observed "
            f"LOCAL_WORLD_SIZE={observed_distributed['local_world_size']}."
        )

    if (
        device_runtime["selected_device_count"] > 0
        and planned_distributed["num_processes_per_machine"]
        != device_runtime["selected_device_count"]
    ):
        messages.append(
            "planned num_processes_per_machine="
            f"{planned_distributed['num_processes_per_machine']} but selected "
            f"device count={device_runtime['selected_device_count']}."
        )

    return messages


def _validate_dataset_input_config(
    config: DictConfig,
) -> None:
    _validate_exclusive_dataset_inputs(
        path_label="dataset",
        dataset_file_path=config.dataset_file_path,
        dataset_file_paths=config.dataset_file_paths,
        dataset_files=config.dataset_files,
    )
    _validate_exclusive_dataset_inputs(
        path_label="val_dataset",
        dataset_file_path=config.val_dataset_file_path,
        dataset_file_paths=config.val_dataset_file_paths,
        dataset_files=config.val_dataset_files,
    )
    _validate_exclusive_dataset_inputs(
        path_label="test_dataset",
        dataset_file_path=config.test_dataset_file_path,
        dataset_file_paths=config.test_dataset_file_paths,
        dataset_files=config.test_dataset_files,
    )

    if (config.dataset_file_paths is not None or config.dataset_files is not None) and (
        config.dataset_mix_name is None or config.dataset_mix_name.strip() == ""
    ):
        raise ValueError(
            "dataset_mix_name is required when dataset_file_paths or dataset_files is set."
        )

    _validate_dataset_resampling_config(config=config)
    _validate_dataset_files_without_weight(
        dataset_files=config.val_dataset_files,
        path_label="val_dataset",
    )
    _validate_dataset_files_without_weight(
        dataset_files=config.test_dataset_files,
        path_label="test_dataset",
    )

    if config.use_validation:
        return

    if (
        config.val_dataset_file_path is None
        and config.val_dataset_file_paths is None
        and config.val_dataset_files is None
    ):
        return

    raise ValueError("val_dataset_file_path(s) require use_validation=true.")


def _validate_exclusive_dataset_inputs(
    path_label: str,
    dataset_file_path: Optional[str],
    dataset_file_paths: Optional[ListConfig],
    dataset_files: Optional[ListConfig],
) -> None:
    enabled_count = sum(
        item is not None
        for item in [
            dataset_file_path,
            dataset_file_paths,
            dataset_files,
        ]
    )
    if enabled_count <= 1:
        return
    raise ValueError(
        f"{path_label}_file_path, {path_label}_file_paths, and {path_label}_files are mutually exclusive."
    )


def _validate_dataset_resampling_config(
    config: DictConfig,
) -> None:
    if config.dataset_resampling.strategy != "weighted_offline":
        raise ValueError("dataset_resampling.strategy must be weighted_offline.")
    if config.dataset_resampling.target_size is not None:
        if not isinstance(config.dataset_resampling.target_size, int):
            raise ValueError(
                "dataset_resampling.target_size must be an integer or null."
            )
        if config.dataset_resampling.target_size <= 0:
            raise ValueError("dataset_resampling.target_size must be positive.")
    if config.dataset_resampling.enabled:
        if config.dataset_files is None:
            raise ValueError("dataset_resampling.enabled=true requires dataset_files.")
        _validate_dataset_files_with_weight(
            dataset_files=config.dataset_files,
            path_label="dataset",
        )
        return
    _validate_dataset_files_without_weight(
        dataset_files=config.dataset_files,
        path_label="dataset",
    )


def _validate_dataset_files_with_weight(
    dataset_files: Optional[ListConfig],
    path_label: str,
) -> None:
    if dataset_files is None:
        return
    for dataset_file in dataset_files:
        if "weight" not in dataset_file:
            raise ValueError(
                f"{path_label}_files weight is required when resampling is enabled."
            )
        if dataset_file.weight is None:
            raise ValueError(
                f"{path_label}_files weight is required when resampling is enabled."
            )
        if dataset_file.weight <= 0:
            raise ValueError(f"{path_label}_files weight must be positive.")


def _validate_dataset_files_without_weight(
    dataset_files: Optional[ListConfig],
    path_label: str,
) -> None:
    if dataset_files is None:
        return
    for dataset_file in dataset_files:
        if "weight" in dataset_file and dataset_file.weight is not None:
            if path_label != "dataset":
                raise ValueError(f"{path_label}_files do not support weight.")
            raise ValueError(
                f"{path_label}_files weight requires dataset_resampling.enabled=true."
            )
