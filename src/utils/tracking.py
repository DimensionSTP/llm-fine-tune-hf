from typing import Dict, Any
import os
import json
import uuid

from omegaconf import DictConfig


def init_wandb_train_tracking(
    config: DictConfig,
) -> None:
    wandb = _import_wandb()
    tracking_run_id = resolve_wandb_train_run_id(config=config)
    wandb.init(
        project=config.project_name,
        name=config.logging_name,
        id=tracking_run_id,
        resume="allow",
    )
    write_tracking_metadata(
        config=config,
        tracking_run_id=tracking_run_id,
    )


def resolve_wandb_train_run_id(
    config: DictConfig,
) -> str:
    validate_tracking_identity_config(config=config)
    metadata = read_tracking_metadata(config=config)
    if config.resume_training:
        tracking_run_id = metadata.get("tracking_run_id")
        if not isinstance(tracking_run_id, str) or tracking_run_id == "":
            raise ValueError(
                "tracking_metadata.json with tracking_run_id is required when "
                "resume_training is true. Refusing to fall back to artifact run_id "
                "because it can merge unrelated W&B runs."
            )
        return tracking_run_id

    if "tracking_run_id" in metadata:
        raise ValueError(
            "tracking_metadata.json already exists for a fresh training run. "
            "Use resume_training=true for interrupted-run resume, or allocate a new "
            "artifact run directory."
        )

    return build_new_tracking_run_id()


def validate_tracking_identity_config(
    config: DictConfig,
) -> None:
    if config.project_name is None:
        raise ValueError("project_name is required for W&B train tracking.")
    if config.logging_name is None:
        raise ValueError("logging_name is required for W&B train tracking.")
    if config.output_base_dir is None:
        raise ValueError("output_base_dir is required for W&B train tracking.")
    if config.output_dir is None:
        raise ValueError("output_dir is required for W&B train tracking.")
    if config.run_id is None:
        raise ValueError("run_id is required for W&B train tracking.")


def build_new_tracking_run_id() -> str:
    return f"run-{uuid.uuid4().hex}"


def write_tracking_metadata(
    config: DictConfig,
    tracking_run_id: str,
) -> None:
    path = get_tracking_metadata_path(config=config)
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )
    write_json(
        path=path,
        payload={
            "backend": "wandb",
            "artifact_run_id": str(config.run_id),
            "tracking_run_id": str(tracking_run_id),
            "project_name": str(config.project_name),
            "logging_name": str(config.logging_name),
            "output_base_dir": str(config.output_base_dir),
            "output_dir": str(config.output_dir),
        },
    )


def read_tracking_metadata(
    config: DictConfig,
) -> Dict[str, Any]:
    path = get_tracking_metadata_path(config=config)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid tracking metadata payload: {path}")
    return payload


def get_tracking_metadata_path(
    config: DictConfig,
) -> str:
    return os.path.join(
        str(config.output_dir),
        "tracking_metadata.json",
    )


def write_json(
    path: str,
    payload: Dict[str, Any],
) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")


def _import_wandb() -> Any:
    try:
        import wandb
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "W&B train tracking requires wandb. Install project dependencies first."
        ) from error
    return wandb
