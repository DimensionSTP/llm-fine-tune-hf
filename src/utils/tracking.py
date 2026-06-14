from typing import Dict, Any
import os
import json
import uuid

from omegaconf import DictConfig

import pandas as pd


def init_train_tracking(
    config: DictConfig,
) -> None:
    backend = config.tracking.backend
    if backend == "wandb":
        _init_wandb_train_tracking(config=config)
        return
    if backend == "mlflow":
        _init_mlflow_train_tracking(config=config)
        return
    raise ValueError(f"Unsupported tracking backend: {backend}.")


def init_eval_tracking(
    config: DictConfig,
) -> None:
    backend = config.tracking.backend
    if backend == "wandb":
        _init_wandb_eval_tracking(config=config)
        return
    if backend == "mlflow":
        _init_mlflow_eval_tracking(config=config)
        return
    raise ValueError(f"Unsupported tracking backend: {backend}.")


def log_tracking_table(
    config: DictConfig,
    key: str,
    dataframe: pd.DataFrame,
) -> None:
    backend = config.tracking.backend
    if backend == "wandb":
        wandb = _import_wandb()
        wandb.log({key: wandb.Table(dataframe=dataframe)})
        return
    if backend == "mlflow":
        mlflow = _import_mlflow()
        mlflow.log_table(
            data=dataframe,
            artifact_file=f"{key}.json",
        )
        return
    raise ValueError(f"Unsupported tracking backend: {backend}.")


def alert_tracking(
    config: DictConfig,
    title: str,
    text: str,
    level: str,
) -> None:
    backend = config.tracking.backend
    if backend == "wandb":
        wandb = _import_wandb()
        if wandb.run is not None:
            wandb.run.alert(
                title=title,
                text=text,
                level=level,
            )
        return
    if backend == "mlflow":
        mlflow = _import_mlflow()
        if mlflow.active_run() is not None:
            mlflow.set_tag(
                key=f"alert.{level.lower()}.{_normalize_tracking_key(value=title)}",
                value=text,
            )
        return
    raise ValueError(f"Unsupported tracking backend: {backend}.")


def finish_tracking(
    config: DictConfig,
) -> None:
    backend = config.tracking.backend
    if backend == "wandb":
        return
    if backend == "mlflow":
        mlflow = _import_mlflow()
        if mlflow.active_run() is not None:
            mlflow.end_run()
        return
    raise ValueError(f"Unsupported tracking backend: {backend}.")


def _init_wandb_train_tracking(
    config: DictConfig,
) -> None:
    wandb = _import_wandb()
    tracking_run_id = _resolve_wandb_train_run_id(config=config)
    wandb.init(
        project=config.project_name,
        name=config.logging_name,
        id=tracking_run_id,
        resume="allow",
    )
    _write_tracking_metadata(
        config=config,
        tracking_run_id=tracking_run_id,
    )


def _resolve_wandb_train_run_id(
    config: DictConfig,
) -> str:
    _validate_tracking_identity_config(config=config)
    metadata = _read_tracking_metadata(config=config)
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

    return _build_new_tracking_run_id()


def _validate_tracking_identity_config(
    config: DictConfig,
) -> None:
    if config.project_name is None:
        raise ValueError("project_name is required for train tracking.")
    if config.logging_name is None:
        raise ValueError("logging_name is required for train tracking.")
    if config.output_base_dir is None:
        raise ValueError("output_base_dir is required for train tracking.")
    if config.output_dir is None:
        raise ValueError("output_dir is required for train tracking.")
    if config.run_id is None:
        raise ValueError("run_id is required for train tracking.")


def _build_new_tracking_run_id() -> str:
    return f"run-{uuid.uuid4().hex}"


def _init_wandb_eval_tracking(
    config: DictConfig,
) -> None:
    wandb = _import_wandb()
    wandb.init(
        project=config.project_name,
        name=config.model_detail,
    )


def _init_mlflow_train_tracking(
    config: DictConfig,
) -> None:
    mlflow = _import_mlflow()
    _validate_tracking_identity_config(config=config)
    _configure_mlflow(config=config)
    metadata = _read_tracking_metadata(config=config)
    tracking_run_id = metadata.get("tracking_run_id")
    if config.resume_training:
        if not isinstance(tracking_run_id, str) or tracking_run_id == "":
            raise ValueError(
                "tracking_metadata.json with tracking_run_id is required when "
                "resume_training is true. Refusing to start a new MLflow run "
                "because it can split a resumed training run across tracking runs."
            )
        active_run = mlflow.start_run(run_id=tracking_run_id)
    else:
        if "tracking_run_id" in metadata:
            raise ValueError(
                "tracking_metadata.json already exists for a fresh training run. "
                "Use resume_training=true for interrupted-run resume, or allocate "
                "a new artifact run directory."
            )
        active_run = mlflow.start_run(
            run_name=config.logging_name,
            tags=_build_mlflow_train_tags(config=config),
        )
    _write_tracking_metadata(
        config=config,
        tracking_run_id=active_run.info.run_id,
    )


def _init_mlflow_eval_tracking(
    config: DictConfig,
) -> None:
    mlflow = _import_mlflow()
    _configure_mlflow(config=config)
    mlflow.start_run(
        run_name=config.model_detail,
        tags=_build_mlflow_eval_tags(config=config),
    )


def _configure_mlflow(
    config: DictConfig,
) -> None:
    mlflow = _import_mlflow()
    if config.tracking.tracking_uri is not None:
        mlflow.set_tracking_uri(config.tracking.tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(config.project_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            name=config.project_name,
            artifact_location=config.tracking.artifact_location,
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)


def _build_mlflow_train_tags(
    config: DictConfig,
) -> Dict[str, str]:
    tags = _build_common_tracking_tags(config=config)
    tags["logging_name"] = str(config.logging_name)
    tags["resume_training"] = str(config.resume_training)
    return tags


def _build_mlflow_eval_tags(
    config: DictConfig,
) -> Dict[str, str]:
    tags = _build_common_tracking_tags(config=config)
    tags["model_detail"] = str(config.model_detail)
    return tags


def _build_common_tracking_tags(
    config: DictConfig,
) -> Dict[str, str]:
    tags = {
        "mode": str(config.mode),
        "project_name": str(config.project_name),
        "fine_tune_method": str(config.fine_tune_method),
        "dataset_name": str(config.dataset_name),
        "model_type": str(config.model_type),
    }
    if config.run_id is not None:
        tags["artifact_run_id"] = str(config.run_id)
    if config.output_dir is not None:
        tags["output_dir"] = str(config.output_dir)
    return tags


def _write_tracking_metadata(
    config: DictConfig,
    tracking_run_id: str,
) -> None:
    path = _get_tracking_metadata_path(config=config)
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )
    _write_json(
        path=path,
        payload={
            "backend": str(config.tracking.backend),
            "artifact_run_id": str(config.run_id),
            "tracking_run_id": str(tracking_run_id),
            "project_name": str(config.project_name),
            "logging_name": str(config.logging_name),
            "output_base_dir": str(config.output_base_dir),
            "output_dir": str(config.output_dir),
        },
    )


def _read_tracking_metadata(
    config: DictConfig,
) -> Dict[str, Any]:
    path = _get_tracking_metadata_path(config=config)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid tracking metadata payload: {path}")
    return payload


def _get_tracking_metadata_path(
    config: DictConfig,
) -> str:
    return os.path.join(
        str(config.output_dir),
        "tracking_metadata.json",
    )


def _write_json(
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


def _normalize_tracking_key(
    value: str,
) -> str:
    return "_".join(value.lower().split())


def _import_wandb() -> Any:
    try:
        import wandb
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "tracking.backend=wandb requires wandb. Install project dependencies first."
        ) from error
    return wandb


def _import_mlflow() -> Any:
    try:
        import mlflow
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "tracking.backend=mlflow requires mlflow. Install project dependencies first."
        ) from error
    return mlflow
