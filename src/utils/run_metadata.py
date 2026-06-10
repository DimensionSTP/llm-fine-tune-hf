from typing import Dict, List, Tuple, Optional, Any
import os
import re
import json
import subprocess
import sys
import time

from omegaconf import DictConfig, ListConfig, OmegaConf

from transformers import TrainingArguments

from ..helpers.dataset_paths import build_dataset_file_path_metadata
from .distributed_runtime import build_distributed_runtime_snapshot


def prepare_train_artifact_config(
    config: DictConfig,
    rank: int,
) -> None:
    if config.mode != "train":
        return

    if config.resume_training:
        prepare_resume_artifact_config(config=config)
        return

    output_base_dir = os.path.normpath(str(config.output_base_dir))
    run_id, output_dir = allocate_or_read_run_directory(
        output_base_dir=output_base_dir,
        rank=rank,
        allocation_timeout_seconds=float(
            config.run_metadata.allocation_timeout_seconds
        ),
        allocation_poll_interval_seconds=float(
            config.run_metadata.allocation_poll_interval_seconds
        ),
        allocation_freshness_grace_seconds=float(
            config.run_metadata.allocation_freshness_grace_seconds
        ),
    )
    config.run_id = run_id
    config.output_dir = str(output_dir)


def prepare_resume_artifact_config(
    config: DictConfig,
) -> None:
    if config.resume_from_checkpoint is None:
        raise ValueError(
            "resume_from_checkpoint is required when resume_training is true."
        )

    resume_from_checkpoint = os.path.normpath(str(config.resume_from_checkpoint))
    if not os.path.exists(resume_from_checkpoint):
        raise ValueError(
            f"resume_from_checkpoint does not exist: {resume_from_checkpoint}"
        )

    output_dir = get_resume_output_dir(
        resume_from_checkpoint=resume_from_checkpoint,
    )
    config.output_base_dir = os.path.dirname(output_dir)
    config.run_id = os.path.basename(output_dir)
    config.output_dir = output_dir


def allocate_or_read_run_directory(
    output_base_dir: str,
    rank: int,
    allocation_timeout_seconds: float,
    allocation_poll_interval_seconds: float,
    allocation_freshness_grace_seconds: float,
) -> Tuple[str, str]:
    allocation_read_started_at = time.time()
    if get_world_size() <= 1:
        return allocate_next_run_directory(output_base_dir=output_base_dir)

    allocation_key = get_allocation_key()
    allocation_path = get_allocation_path(
        output_base_dir=output_base_dir,
        allocation_key=allocation_key,
    )
    if rank == 0:
        run_id, output_dir = allocate_next_run_directory(
            output_base_dir=output_base_dir,
        )
        write_json(
            path=allocation_path,
            payload={
                "allocation_key": allocation_key,
                "output_base_dir": output_base_dir,
                "run_id": run_id,
                "output_dir": str(output_dir),
                "created_at": time.time(),
            },
        )
        return run_id, output_dir

    return read_run_directory_allocation(
        allocation_path=allocation_path,
        allocation_key=allocation_key,
        output_base_dir=output_base_dir,
        allocation_read_started_at=allocation_read_started_at,
        allocation_timeout_seconds=allocation_timeout_seconds,
        allocation_poll_interval_seconds=allocation_poll_interval_seconds,
        allocation_freshness_grace_seconds=allocation_freshness_grace_seconds,
    )


def allocate_next_run_directory(
    output_base_dir: str,
) -> Tuple[str, str]:
    os.makedirs(
        output_base_dir,
        exist_ok=True,
    )
    next_index = get_next_run_index(output_base_dir=output_base_dir)
    while True:
        run_id = f"run-{next_index:04d}"
        output_dir = os.path.join(
            output_base_dir,
            run_id,
        )
        try:
            os.mkdir(output_dir)
        except FileExistsError:
            next_index += 1
            continue
        return run_id, output_dir


def read_run_directory_allocation(
    allocation_path: str,
    allocation_key: str,
    output_base_dir: str,
    allocation_read_started_at: float,
    allocation_timeout_seconds: float,
    allocation_poll_interval_seconds: float,
    allocation_freshness_grace_seconds: float,
) -> Tuple[str, str]:
    if allocation_timeout_seconds <= 0:
        raise ValueError("allocation_timeout_seconds must be greater than 0.")
    if allocation_poll_interval_seconds <= 0:
        raise ValueError("allocation_poll_interval_seconds must be greater than 0.")
    if allocation_freshness_grace_seconds < 0:
        raise ValueError(
            "allocation_freshness_grace_seconds must be greater than or equal to 0."
        )

    allocation_deadline_at = time.monotonic() + allocation_timeout_seconds
    while time.monotonic() < allocation_deadline_at:
        if os.path.isfile(allocation_path):
            try:
                with open(allocation_path, encoding="utf-8") as file:
                    payload = json.load(file)
            except json.JSONDecodeError:
                time.sleep(allocation_poll_interval_seconds)
                continue
            if is_current_run_directory_allocation(
                payload=payload,
                allocation_key=allocation_key,
                output_base_dir=output_base_dir,
                allocation_read_started_at=allocation_read_started_at,
                allocation_freshness_grace_seconds=allocation_freshness_grace_seconds,
            ):
                return str(payload["run_id"]), str(payload["output_dir"])
        time.sleep(allocation_poll_interval_seconds)

    raise TimeoutError(f"Timed out waiting for run allocation: {allocation_path}")


def is_current_run_directory_allocation(
    payload: Any,
    allocation_key: str,
    output_base_dir: str,
    allocation_read_started_at: float,
    allocation_freshness_grace_seconds: float,
) -> bool:
    if not isinstance(payload, dict):
        return False

    run_id = payload.get("run_id")
    output_dir = payload.get("output_dir")
    payload_output_base_dir = payload.get("output_base_dir")
    created_at = payload.get("created_at")
    if not isinstance(run_id, str):
        return False
    if not isinstance(output_dir, str):
        return False
    if not isinstance(payload_output_base_dir, str):
        return False
    if not isinstance(created_at, (int, float)):
        return False
    if payload.get("allocation_key") != allocation_key:
        return False
    if os.path.normpath(payload_output_base_dir) != os.path.normpath(output_base_dir):
        return False
    if created_at < allocation_read_started_at - allocation_freshness_grace_seconds:
        return False
    if not os.path.isdir(output_dir):
        return False
    return os.path.basename(output_dir) == run_id


def get_next_run_index(
    output_base_dir: str,
) -> int:
    run_indices = [
        int(match.group(1))
        for child_name in os.listdir(output_base_dir)
        for child_path in [
            os.path.join(
                output_base_dir,
                child_name,
            )
        ]
        if os.path.isdir(child_path)
        for match in [re.match(r"^run-([0-9]{4})$", child_name)]
        if match is not None
    ]
    if len(run_indices) == 0:
        return 1
    return max(run_indices) + 1


def get_resume_output_dir(
    resume_from_checkpoint: str,
) -> str:
    if re.match(r"^checkpoint-[0-9]+$", os.path.basename(resume_from_checkpoint)):
        return os.path.dirname(resume_from_checkpoint)
    return resume_from_checkpoint


def get_allocation_path(
    output_base_dir: str,
    allocation_key: str,
) -> str:
    allocation_dir = os.path.join(
        os.path.dirname(output_base_dir),
        ".run_allocations",
        os.path.basename(output_base_dir),
    )
    os.makedirs(
        allocation_dir,
        exist_ok=True,
    )
    return os.path.join(
        allocation_dir,
        f"{allocation_key}.json",
    )


def get_allocation_key() -> str:
    raw_key = "-".join(
        [
            os.environ.get("TORCHELASTIC_RUN_ID", "none"),
            os.environ.get("MASTER_ADDR", "none"),
            os.environ.get("MASTER_PORT", "none"),
            os.environ.get("WORLD_SIZE", "1"),
        ]
    )
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_key)


def write_run_metadata(
    config: DictConfig,
    training_arguments: TrainingArguments,
    rank: int,
) -> None:
    if rank != 0:
        return

    output_dir = str(config.output_dir)
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    training_arguments_payload = json.loads(
        training_arguments.to_json_string(),
    )
    resolved_config = OmegaConf.to_container(
        config,
        resolve=True,
    )

    write_json(
        path=os.path.join(
            output_dir,
            "run_manifest.json",
        ),
        payload=build_run_manifest(
            config=config,
            training_arguments=training_arguments_payload,
            resolved_config=resolved_config,
        ),
    )

    with open(
        os.path.join(
            output_dir,
            "resolved_config.yaml",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            OmegaConf.to_yaml(
                config,
                resolve=True,
            )
        )

    write_json(
        path=os.path.join(
            output_dir,
            "training_args.json",
        ),
        payload=training_arguments_payload,
    )


def build_run_manifest(
    config: DictConfig,
    training_arguments: Dict[str, Any],
    resolved_config: Any,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "run": build_run_section(config=config),
        "paths": build_paths_section(config=config),
        "source": build_source_section(),
        "runtime": build_runtime_section(config=config),
        "summary": build_summary_section(config=config),
        "method_hyperparameters": build_method_hyperparameters(config=config),
        "training_arguments": training_arguments,
        "resolved_config": resolved_config,
    }


def build_run_section(
    config: DictConfig,
) -> Dict[str, Any]:
    return {
        "run_id": config.run_id,
        "fine_tune_method": config.fine_tune_method,
        "model_name": config.model_name,
        "dataset_name": config.dataset_name,
        "strategy": config.strategy,
        "logging_name": config.logging_name,
        "run_name": config.run_name,
    }


def build_paths_section(
    config: DictConfig,
) -> Dict[str, Any]:
    return {
        "output_base_dir": config.output_base_dir,
        "output_dir": config.output_dir,
        "save_detail": config.save_detail,
        "dataset": build_dataset_file_path_metadata(
            dataset_name=config.dataset_name,
            dataset_format=config.dataset_format,
            data_path=config.data_path,
            dataset_subdir=config.dataset_subdir,
            dataset_file_path=config.dataset_file_path,
            allow_dataset_file_name_mismatch=config.allow_dataset_file_name_mismatch,
        ),
        "test_dataset": build_dataset_file_path_metadata(
            dataset_name=config.dataset_name,
            dataset_format=config.dataset_format,
            data_path=config.data_path,
            dataset_subdir=config.test_dataset_subdir,
            dataset_file_path=config.test_dataset_file_path,
            allow_dataset_file_name_mismatch=config.allow_test_dataset_file_name_mismatch,
        ),
    }


def build_source_section() -> Dict[str, Any]:
    return {
        "git_revision": get_git_revision(),
        "python_argv": sys.argv,
        "working_directory": os.getcwd(),
    }


def build_runtime_section(
    config: DictConfig,
) -> Dict[str, Any]:
    return build_distributed_runtime_snapshot(config=config)


def build_summary_section(
    config: DictConfig,
) -> Dict[str, Any]:
    return select_config_values(
        config=config,
        paths=[
            "run_id",
            "output_base_dir",
            "output_dir",
            "model_name",
            "fine_tune_method",
            "model_type",
            "model_detail",
            "pretrained_model_name",
            "revision",
            "dataset_name",
            "dataset_format",
            "data_path",
            "dataset_subdir",
            "dataset_file_path",
            "allow_dataset_file_name_mismatch",
            "test_dataset_subdir",
            "test_dataset_file_path",
            "allow_test_dataset_file_name_mismatch",
            "dataset_image",
            "data_type",
            "split",
            "split_ratio",
            "is_strict_split",
            "seed",
            "strategy",
            "save_detail",
            "batch_size",
            "eval_batch_size",
            "gradient_accumulation_steps",
            "devices",
            "distributed",
            "workers_ratio",
            "use_all_workers",
            "lr",
            "weight_decay",
            "scheduler_type",
            "warmup_ratio",
            "epoch",
            "step",
            "optim",
            "max_grad_norm",
            "precision",
            "is_bf16",
            "is_quantized",
            "quantization_config",
            "model_loading",
            "is_peft",
            "peft_config",
            "gradient_checkpointing",
            "gradient_checkpointing_kwargs",
            "max_length",
            "max_new_tokens",
            "left_padding",
            "is_enable_thinking",
            "chat_template_path",
            "logging_name",
            "run_name",
        ],
        extra={
            "effective_batch_size": get_effective_batch_size(config=config),
        },
    )


def build_method_hyperparameters(
    config: DictConfig,
) -> Dict[str, Any]:
    common_paths_by_method: Dict[str, List[str]] = {
        "sft": [
            "sft_loss_type",
            "truncation_mode",
            "pad_to_multiple_of",
            "response_start_template",
            "response_end_template",
            "chat_template_path",
            "training_arguments.use_liger_kernel",
            "training_arguments.packing",
        ],
        "dpo": [
            "beta",
            "truncation_mode",
            "pad_to_multiple_of",
            "training_arguments.use_liger_kernel",
        ],
        "kto": [
            "beta",
            "training_arguments.use_liger_kernel",
        ],
        "gkd": [
            "teacher.upload_user",
            "teacher.model_type",
            "teacher.model",
            "teacher.model_init_kwargs",
            "max_new_tokens",
            "generation_config.temperature",
            "truncation_mode",
            "pad_to_multiple_of",
            "training_arguments.use_liger_kernel",
            "training_arguments.packing",
        ],
        "grpo": [
            "loss_type",
            "beta",
            "epsilon",
            "epsilon_high",
            "num_generations",
            "importance_sampling_level",
            "scale_rewards",
            "generation_config.temperature",
            "generation_config.top_p",
            "generation_config.top_k",
            "log_completions",
            "use_vllm",
            "vllm_mode",
            "vllm_tensor_parallel_size",
            "gpu_memory_utilization",
            "sapo_temperature_pos",
            "sapo_temperature_neg",
            "completion_termination",
            "reward",
            "reward_database",
            "reward_embedding",
        ],
        "sdpo": [
            "loss_type",
            "beta",
            "epsilon",
            "epsilon_high",
            "num_generations",
            "importance_sampling_level",
            "scale_rewards",
            "generation_config.temperature",
            "generation_config.top_p",
            "generation_config.top_k",
            "use_vllm",
            "vllm_mode",
            "vllm_tensor_parallel_size",
            "gpu_memory_utilization",
            "distillation_weight",
            "teacher_regularization",
            "ema_update_rate",
            "use_successful_as_teacher",
            "success_reward_threshold",
            "reward",
            "reward_database",
            "reward_embedding",
        ],
        "async_grpo": [
            "epsilon",
            "epsilon_high",
            "num_generations",
            "generation_config.temperature",
            "log_completions",
            "num_completions_to_print",
            "vllm_server_base_url",
            "vllm_server_timeout",
            "request_timeout",
            "max_inflight_tasks",
            "max_staleness",
            "queue_maxsize",
            "weight_sync_steps",
            "async_runtime",
            "reward",
            "reward_database",
            "reward_embedding",
        ],
    }
    return select_config_values(
        config=config,
        paths=common_paths_by_method[str(config.fine_tune_method)],
    )


def select_config_values(
    config: DictConfig,
    paths: List[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    selected = dict(extra or {})
    sentinel = object()
    for path in paths:
        value = OmegaConf.select(
            config,
            path,
            default=sentinel,
        )
        if value is not sentinel:
            selected[path] = _to_jsonable(value)
    return selected


def get_effective_batch_size(
    config: DictConfig,
) -> int:
    runtime_snapshot = build_distributed_runtime_snapshot(config=config)
    return int(runtime_snapshot["effective_batch_size"])


def get_device_count(
    config: DictConfig,
) -> int:
    devices = config.devices
    if isinstance(devices, int):
        return devices
    if isinstance(devices, str):
        return len([device for device in devices.split(",") if device])
    if isinstance(devices, (list, ListConfig)):
        return len(devices)
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_git_revision() -> Optional[str]:
    repo_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
        )
    )
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "--short",
                "HEAD",
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    return result.stdout.strip()


def write_json(
    path: str,
    payload: Dict[str, Any],
) -> None:
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )
    temp_path = f"{path}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(
            _to_jsonable(payload),
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
    os.replace(
        temp_path,
        path,
    )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(
            value,
            resolve=True,
        )
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
