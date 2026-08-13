from typing import Dict, Tuple, Optional, Any
import os
import re
import json
import logging
import subprocess
import sys
import time

from omegaconf import DictConfig, ListConfig, OmegaConf

from transformers import TrainingArguments

from ..helpers.dataset_paths import build_train_dataset_input_metadata
from .distributed_runtime import build_distributed_runtime_snapshot
from .dataloader_runtime import resolve_dataloader_runtime
from .peft_initialization import build_peft_initialization_metadata


def prepare_train_artifact_config(
    config: DictConfig,
    rank: int,
) -> None:
    if config.mode != "train":
        return
    if config.memory_preflight.is_probe:
        return

    if config.resume_training:
        _prepare_resume_artifact_config(config=config)
        return

    output_base_dir = os.path.normpath(str(config.output_base_dir))
    run_id, output_dir = _allocate_or_read_run_directory(
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


def write_run_metadata(
    config: DictConfig,
    rank: int,
) -> None:
    if rank != 0 or config.memory_preflight.is_probe:
        return

    output_dir = str(config.output_dir)
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    manifest_path = _get_run_manifest_path(config=config)
    manifest = {
        "run_id": str(config.run_id),
        "status": "prepared",
        "stage": "setup",
        "artifacts": {},
        "source": _build_source_section(),
    }
    _write_json(
        path=manifest_path,
        payload=manifest,
    )
    _write_resolved_config(config=config)
    manifest["artifacts"] = _build_train_artifact_section(config=config)
    _write_json(
        path=manifest_path,
        payload=manifest,
    )


def write_training_metadata(
    config: DictConfig,
    training_arguments: TrainingArguments,
    rank: int,
) -> None:
    if rank != 0 or config.memory_preflight.is_probe:
        return

    training_arguments_payload = json.loads(
        training_arguments.to_json_string(),
    )
    _write_resolved_config(config=config)
    _write_json(
        path=os.path.join(
            str(config.output_dir),
            "training_args.json",
        ),
        payload=training_arguments_payload,
    )

    manifest = _read_run_manifest(config=config)
    manifest["artifacts"] = _build_train_artifact_section(config=config)
    manifest["inputs"] = _build_input_section(config=config)
    manifest["runtime"] = _build_runtime_section(config=config)
    peft_initialization = build_peft_initialization_metadata(
        config=config,
    )
    if len(peft_initialization) > 0:
        manifest["peft_initialization"] = peft_initialization
    else:
        manifest.pop(
            "peft_initialization",
            None,
        )
    _write_json(
        path=_get_run_manifest_path(config=config),
        payload=manifest,
    )


def write_vision_patch_embedding_metadata(
    config: DictConfig,
    compatibility_result: Dict[str, Any],
    rank: int,
) -> None:
    if rank != 0 or config.memory_preflight.is_probe:
        return

    manifest = _read_run_manifest(config=config)
    manifest["runtime"]["vision_patch_embedding"] = compatibility_result
    _write_json(
        path=_get_run_manifest_path(config=config),
        payload=manifest,
    )


def update_run_metadata(
    config: DictConfig,
    status: str,
    stage: str,
    error: Optional[BaseException],
    rank: int,
) -> None:
    if rank != 0 or config.memory_preflight.is_probe:
        return
    if status not in {"prepared", "running", "completed", "failed", "interrupted"}:
        raise ValueError(f"Unsupported run metadata status: {status}.")
    if stage not in {"preflight", "setup", "training", "saving", "completed"}:
        raise ValueError(f"Unsupported run metadata stage: {stage}.")
    if status in {"failed", "interrupted"} and error is None:
        raise ValueError(f"Run metadata status {status} requires an error.")
    if status not in {"failed", "interrupted"} and error is not None:
        raise ValueError(f"Run metadata status {status} must not include an error.")

    manifest = _read_run_manifest(config=config)
    manifest["status"] = status
    manifest["stage"] = stage
    manifest["artifacts"] = _build_train_artifact_section(config=config)
    if error is None:
        manifest.pop(
            "failure",
            None,
        )
    else:
        manifest["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    _write_json(
        path=_get_run_manifest_path(config=config),
        payload=manifest,
    )


def update_run_metadata_preserving_error(
    config: DictConfig,
    status: str,
    stage: str,
    error: BaseException,
    rank: int,
) -> None:
    try:
        update_run_metadata(
            config=config,
            status=status,
            stage=stage,
            error=error,
            rank=rank,
        )
    except Exception as metadata_error:
        logging.getLogger(__name__).exception(
            f"Failed to update run metadata while preserving the pipeline error: {metadata_error}"
        )


def _prepare_resume_artifact_config(
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

    output_dir = _get_resume_output_dir(
        resume_from_checkpoint=resume_from_checkpoint,
    )
    config.output_base_dir = os.path.dirname(output_dir)
    config.run_id = os.path.basename(output_dir)
    config.output_dir = output_dir


def _allocate_or_read_run_directory(
    output_base_dir: str,
    rank: int,
    allocation_timeout_seconds: float,
    allocation_poll_interval_seconds: float,
    allocation_freshness_grace_seconds: float,
) -> Tuple[str, str]:
    allocation_read_started_at = time.time()
    if _get_world_size() <= 1:
        return _allocate_next_run_directory(output_base_dir=output_base_dir)

    allocation_key = _get_allocation_key()
    allocation_path = _get_allocation_path(
        output_base_dir=output_base_dir,
        allocation_key=allocation_key,
    )
    if rank == 0:
        run_id, output_dir = _allocate_next_run_directory(
            output_base_dir=output_base_dir,
        )
        _write_json(
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

    return _read_run_directory_allocation(
        allocation_path=allocation_path,
        allocation_key=allocation_key,
        output_base_dir=output_base_dir,
        allocation_read_started_at=allocation_read_started_at,
        allocation_timeout_seconds=allocation_timeout_seconds,
        allocation_poll_interval_seconds=allocation_poll_interval_seconds,
        allocation_freshness_grace_seconds=allocation_freshness_grace_seconds,
    )


def _allocate_next_run_directory(
    output_base_dir: str,
) -> Tuple[str, str]:
    os.makedirs(
        output_base_dir,
        exist_ok=True,
    )
    next_index = _get_next_run_index(output_base_dir=output_base_dir)
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


def _read_run_directory_allocation(
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
                with open(
                    allocation_path,
                    encoding="utf-8",
                ) as file:
                    payload = json.load(file)
            except json.JSONDecodeError:
                time.sleep(allocation_poll_interval_seconds)
                continue
            if _is_current_run_directory_allocation(
                payload=payload,
                allocation_key=allocation_key,
                output_base_dir=output_base_dir,
                allocation_read_started_at=allocation_read_started_at,
                allocation_freshness_grace_seconds=allocation_freshness_grace_seconds,
            ):
                return str(payload["run_id"]), str(payload["output_dir"])
        time.sleep(allocation_poll_interval_seconds)

    raise TimeoutError(f"Timed out waiting for run allocation: {allocation_path}")


def _is_current_run_directory_allocation(
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


def _get_next_run_index(
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
        for match in [
            re.match(
                r"^run-([0-9]{4})$",
                child_name,
            )
        ]
        if match is not None
    ]
    if len(run_indices) == 0:
        return 1
    return max(run_indices) + 1


def _get_resume_output_dir(
    resume_from_checkpoint: str,
) -> str:
    if re.match(r"^checkpoint-[0-9]+$", os.path.basename(resume_from_checkpoint)):
        return os.path.dirname(resume_from_checkpoint)
    return resume_from_checkpoint


def _get_allocation_path(
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


def _get_allocation_key() -> str:
    raw_key = "-".join(
        [
            os.environ.get(
                "TORCHELASTIC_RUN_ID",
                "none",
            ),
            os.environ.get(
                "MASTER_ADDR",
                "none",
            ),
            os.environ.get(
                "MASTER_PORT",
                "none",
            ),
            os.environ.get(
                "WORLD_SIZE",
                "1",
            ),
        ]
    )
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        raw_key,
    )


def _build_train_artifact_section(
    config: DictConfig,
) -> Dict[str, Any]:
    output_dir = str(config.output_dir)
    relative_paths = {
        "resolved_config": "resolved_config.yaml",
        "training_arguments": "training_args.json",
        "tracking": "tracking_metadata.json",
    }
    artifacts = {
        artifact_name: relative_path
        for artifact_name, relative_path in relative_paths.items()
        if os.path.isfile(
            os.path.join(
                output_dir,
                relative_path,
            )
        )
    }
    memory_preflight_artifacts = _build_memory_preflight_artifact_section(
        output_dir=output_dir,
    )
    if len(memory_preflight_artifacts) > 0:
        artifacts["memory_preflight"] = memory_preflight_artifacts
    return artifacts


def _build_memory_preflight_artifact_section(
    output_dir: str,
) -> Dict[str, str]:
    relative_paths = {
        "command": os.path.join(
            "memory_preflight",
            "command.json",
        ),
        "selected_indices": os.path.join(
            "memory_preflight",
            "selected_indices.json",
        ),
        "result": os.path.join(
            "memory_preflight",
            "result.json",
        ),
    }
    return {
        artifact_name: relative_path
        for artifact_name, relative_path in relative_paths.items()
        if os.path.isfile(
            os.path.join(
                output_dir,
                relative_path,
            )
        )
    }


def _build_input_section(
    config: DictConfig,
) -> Dict[str, Any]:
    return build_train_dataset_input_metadata(
        dataset_name=config.dataset_name,
        dataset_format=config.dataset_format,
        data_path=config.data_path,
        dataset_subdir=config.dataset_subdir,
        dataset_file_path=config.dataset_file_path,
        dataset_file_paths=config.dataset_file_paths,
        dataset_files=config.dataset_files,
        allow_dataset_file_name_mismatch=config.allow_dataset_file_name_mismatch,
        val_dataset_file_path=config.val_dataset_file_path,
        val_dataset_file_paths=config.val_dataset_file_paths,
        val_dataset_files=config.val_dataset_files,
        allow_val_dataset_file_name_mismatch=config.allow_val_dataset_file_name_mismatch,
        use_validation=config.use_validation,
        dataset_resampling=config.dataset_resampling,
    )


def _build_source_section() -> Dict[str, Any]:
    return {
        "git_revision": _get_git_revision(),
        "python_argv": sys.argv,
        "working_directory": os.getcwd(),
    }


def _build_runtime_section(
    config: DictConfig,
) -> Dict[str, Any]:
    runtime_snapshot = build_distributed_runtime_snapshot(config=config)
    runtime_snapshot["dataloader_runtime"] = resolve_dataloader_runtime(
        config=config,
        distributed_runtime_snapshot=runtime_snapshot,
    )
    return runtime_snapshot


def _write_resolved_config(
    config: DictConfig,
) -> None:
    path = os.path.join(
        str(config.output_dir),
        "resolved_config.yaml",
    )
    temp_path = f"{path}.tmp.{os.getpid()}"
    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            OmegaConf.to_yaml(
                config,
                resolve=True,
            )
        )
    os.replace(
        temp_path,
        path,
    )


def _read_run_manifest(
    config: DictConfig,
) -> Dict[str, Any]:
    manifest_path = _get_run_manifest_path(config=config)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    with open(
        manifest_path,
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError("run_manifest.json must contain a JSON object.")
    return manifest


def _get_run_manifest_path(
    config: DictConfig,
) -> str:
    return os.path.join(
        str(config.output_dir),
        "run_manifest.json",
    )


def _get_world_size() -> int:
    return int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )


def _get_git_revision() -> Optional[str]:
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


def _write_json(
    path: str,
    payload: Dict[str, Any],
) -> None:
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )
    temp_path = f"{path}.tmp.{os.getpid()}"
    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as file:
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
