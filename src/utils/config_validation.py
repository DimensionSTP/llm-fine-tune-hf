from typing import Dict, List, Optional, Any
import os

from omegaconf import DictConfig

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
    if config.dataset_file_path is not None and config.dataset_file_paths is not None:
        raise ValueError(
            "dataset_file_path and dataset_file_paths are mutually exclusive."
        )

    if (
        config.val_dataset_file_path is not None
        and config.val_dataset_file_paths is not None
    ):
        raise ValueError(
            "val_dataset_file_path and val_dataset_file_paths are mutually exclusive."
        )

    if (
        config.test_dataset_file_path is not None
        and config.test_dataset_file_paths is not None
    ):
        raise ValueError(
            "test_dataset_file_path and test_dataset_file_paths are mutually exclusive."
        )

    if config.dataset_file_paths is not None and (
        config.dataset_mix_name is None or config.dataset_mix_name.strip() == ""
    ):
        raise ValueError("dataset_mix_name is required when dataset_file_paths is set.")

    if config.use_validation:
        return

    if config.val_dataset_file_path is None and config.val_dataset_file_paths is None:
        return

    raise ValueError("val_dataset_file_path(s) require use_validation=true.")
