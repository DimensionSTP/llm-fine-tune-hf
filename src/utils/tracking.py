from typing import Dict, List, Callable, Iterator, Any
import os
import base64
from collections.abc import Collection
from contextlib import contextmanager
from functools import partial, partialmethod, update_wrapper
import io
import json
import logging
import uuid

from omegaconf import DictConfig

import pandas as pd

from transformers import TrainerCallback


def tracking_lifecycle(
    function: Callable[[DictConfig], None],
) -> Callable[[DictConfig], None]:
    return update_wrapper(
        wrapper=partial(
            _run_tracking_lifecycle,
            function=function,
        ),
        wrapped=function,
    )


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


def attach_train_completion_tracking(
    config: DictConfig,
    trainer: Any,
) -> None:
    if (
        config.fine_tune_method != "grpo"
        or not config.log_completions
        or config.tracking.backend != "mlflow"
        or config.memory_preflight.is_probe
        or not _is_tracking_owner()
    ):
        return
    trainer.add_callback(
        _MLflowCompletionArtifactCallback(
            config=config,
            completion_logs=trainer._logs,
        ),
    )


def get_tracking_context(
    config: DictConfig,
) -> Dict[str, Any]:
    backend = config.tracking.backend
    if backend == "wandb":
        wandb = _import_wandb()
        if wandb.run is None:
            return {
                "backend": backend,
                "tracking_run_id": None,
                "run_url": None,
            }
        return {
            "backend": backend,
            "tracking_run_id": wandb.run.id,
            "run_url": wandb.run.url,
        }
    if backend == "mlflow":
        mlflow = _import_mlflow()
        active_run = mlflow.active_run()
        if active_run is None:
            return {
                "backend": backend,
                "tracking_run_id": None,
                "run_url": None,
            }
        tracking_uri = str(mlflow.get_tracking_uri())
        run_url = None
        if tracking_uri.startswith(("http://", "https://")):
            run_url = (
                f"{tracking_uri.rstrip('/')}"
                f"/#/experiments/{active_run.info.experiment_id}"
                f"/runs/{active_run.info.run_id}"
            )
        return {
            "backend": backend,
            "tracking_run_id": active_run.info.run_id,
            "run_url": run_url,
        }
    raise ValueError(f"Unsupported tracking backend: {backend}.")


def finish_tracking(
    config: DictConfig,
    status: str,
) -> None:
    if status not in {"FINISHED", "FAILED", "KILLED"}:
        raise ValueError(f"Unsupported terminal tracking status: {status}.")

    backend = config.tracking.backend
    if backend == "wandb":
        return
    if backend == "mlflow":
        mlflow = _import_mlflow()
        if mlflow.active_run() is not None:
            mlflow.end_run(status=status)
        return
    raise ValueError(f"Unsupported tracking backend: {backend}.")


def _run_tracking_lifecycle(
    config: DictConfig,
    function: Callable[[DictConfig], None],
) -> None:
    if not _is_tracking_owner():
        return function(config=config)

    try:
        result = function(config=config)
    except (KeyboardInterrupt, SystemExit):
        _finish_tracking_preserving_error(
            config=config,
            status="KILLED",
        )
        raise
    except Exception:
        _finish_tracking_preserving_error(
            config=config,
            status="FAILED",
        )
        raise

    finish_tracking(
        config=config,
        status="FINISHED",
    )
    return result


class _MLflowCompletionArtifactCallback(TrainerCallback):
    def __init__(
        self,
        config: DictConfig,
        completion_logs: Dict[str, Any],
    ) -> None:
        self.config = config
        self.completion_logs = completion_logs

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        _log_mlflow_completion_artifacts(
            config=self.config,
            completion_logs=self.completion_logs,
            global_step=state.global_step,
            log_index=len(state.log_history),
        )


def _log_mlflow_completion_artifacts(
    config: DictConfig,
    completion_logs: Dict[str, Any],
    global_step: int,
    log_index: int,
) -> None:
    prompts = completion_logs["prompt"]
    if len(prompts) == 0:
        return

    table = {
        "step": [global_step] * len(prompts),
        "prompt": prompts,
        "completion": completion_logs["completion"],
        **completion_logs["rewards"],
        **completion_logs["extra"],
        "advantage": completion_logs["advantages"],
    }
    if config.log_multimodal:
        table["image_artifacts"] = _log_mlflow_completion_images(
            image_rows=completion_logs["images"],
            num_rows=len(prompts),
            global_step=global_step,
            log_index=log_index,
        )

    mlflow = _import_mlflow()
    mlflow.log_table(
        data=pd.DataFrame(table),
        artifact_file=f"completions/tables/step_{global_step:05d}_log_{log_index:05d}.json",
    )


def _log_mlflow_completion_images(
    image_rows: Collection[Any],
    num_rows: int,
    global_step: int,
    log_index: int,
) -> List[str]:
    if len(image_rows) == 0:
        return ["[]"] * num_rows
    if len(image_rows) != num_rows:
        raise ValueError("GRPO completion image rows must match completion table rows.")

    mlflow = _import_mlflow()
    row_artifact_paths = []
    for row_index, images in enumerate(image_rows):
        artifact_paths = []
        for image_index, image in enumerate(images or []):
            artifact_path = (
                "completions/images/"
                f"step_{global_step:05d}_log_{log_index:05d}_"
                f"row_{row_index:05d}_image_{image_index:03d}.png"
            )
            mlflow.log_image(
                image=mlflow.Image(_resolve_mlflow_completion_image(image=image)),
                artifact_file=artifact_path,
                synchronous=True,
            )
            artifact_paths.append(artifact_path)
        row_artifact_paths.append(
            json.dumps(
                artifact_paths,
                ensure_ascii=False,
            )
        )
    return row_artifact_paths


def _resolve_mlflow_completion_image(
    image: Any,
) -> Any:
    if not isinstance(image, str) or os.path.isfile(image):
        return image

    encoded_image = (
        image.split(
            ",",
            1,
        )[1]
        if image.startswith("data:")
        else image
    )
    image_bytes = base64.b64decode(
        encoded_image,
        validate=True,
    )

    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as decoded_image:
        return decoded_image.copy()


def _finish_tracking_preserving_error(
    config: DictConfig,
    status: str,
) -> None:
    try:
        finish_tracking(
            config=config,
            status=status,
        )
    except Exception as error:
        logging.getLogger(__name__).exception(
            f"Failed to finalize tracking status {status} while preserving the pipeline error: {error}"
        )


def _is_tracking_owner() -> bool:
    return (
        int(
            os.environ.get(
                "RANK",
                0,
            )
        )
        == 0
    )


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
    _validate_tracking_metadata_backend(
        config=config,
        metadata=metadata,
    )
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
    _validate_tracking_metadata_backend(
        config=config,
        metadata=metadata,
    )
    tracking_run_id = metadata.get("tracking_run_id")
    if config.resume_training:
        if not isinstance(tracking_run_id, str) or tracking_run_id == "":
            raise ValueError(
                "tracking_metadata.json with tracking_run_id is required when "
                "resume_training is true. Refusing to start a new MLflow run "
                "because it can split a resumed training run across tracking runs."
            )
        os.environ["MLFLOW_MAX_LOG_PARAMS"] = "0"
        with _patch_mlflow_visible_gpu_monitor(
            enabled=config.tracking.system_metrics.enabled,
        ):
            active_run = mlflow.start_run(
                run_id=tracking_run_id,
                log_system_metrics=config.tracking.system_metrics.enabled,
            )
    else:
        if "tracking_run_id" in metadata:
            raise ValueError(
                "tracking_metadata.json already exists for a fresh training run. "
                "Use resume_training=true for interrupted-run resume, or allocate "
                "a new artifact run directory."
            )
        with _patch_mlflow_visible_gpu_monitor(
            enabled=config.tracking.system_metrics.enabled,
        ):
            active_run = mlflow.start_run(
                run_name=config.logging_name,
                tags=_build_mlflow_train_tags(config=config),
                log_system_metrics=config.tracking.system_metrics.enabled,
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
    with _patch_mlflow_visible_gpu_monitor(
        enabled=config.tracking.system_metrics.enabled,
    ):
        mlflow.start_run(
            run_name=config.model_detail,
            tags=_build_mlflow_eval_tags(config=config),
            log_system_metrics=config.tracking.system_metrics.enabled,
        )


def _configure_mlflow(
    config: DictConfig,
) -> None:
    mlflow = _import_mlflow()
    tracking_uri = config.tracking.tracking_uri
    if config.tracking.require_tracking_uri and (
        tracking_uri is None or str(tracking_uri).strip() == ""
    ):
        raise ValueError(
            "tracking.tracking_uri is required for the selected MLflow server profile."
        )
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    _configure_mlflow_system_metrics(config=config)
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


def _configure_mlflow_system_metrics(
    config: DictConfig,
) -> None:
    if not config.tracking.system_metrics.enabled:
        return
    mlflow = _import_mlflow()
    mlflow.set_system_metrics_sampling_interval(
        config.tracking.system_metrics.sampling_interval_seconds,
    )
    mlflow.set_system_metrics_samples_before_logging(
        config.tracking.system_metrics.samples_before_logging,
    )


@contextmanager
def _patch_mlflow_visible_gpu_monitor(
    enabled: bool,
) -> Iterator[None]:
    if not enabled:
        yield
        return

    visible_gpu_identifiers = _resolve_visible_gpu_identifiers()
    if visible_gpu_identifiers is None:
        yield
        return

    from mlflow.system_metrics.metrics.gpu_monitor import GPUMonitor

    original_init = GPUMonitor.__init__
    GPUMonitor.__init__ = partialmethod(
        _initialize_visible_gpu_monitor,
        original_init=original_init,
    )
    try:
        yield
    finally:
        GPUMonitor.__init__ = original_init


def _initialize_visible_gpu_monitor(
    monitor: Any,
    original_init: Callable[[Any], None],
) -> None:
    original_init(monitor)
    visible_gpu_identifiers = _resolve_visible_gpu_identifiers()
    if visible_gpu_identifiers is None:
        return

    import pynvml

    monitor.gpu_handles = [
        _resolve_visible_gpu_handle(
            gpu_identifier=gpu_identifier,
            pynvml=pynvml,
        )
        for gpu_identifier in visible_gpu_identifiers
    ]
    monitor.num_gpus = len(monitor.gpu_handles)


def _resolve_visible_gpu_identifiers() -> List[str] | None:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is None:
        return None
    if cuda_visible_devices.strip() == "-1":
        return []
    return [
        gpu_identifier.strip()
        for gpu_identifier in cuda_visible_devices.split(",")
        if gpu_identifier.strip()
    ]


def _resolve_visible_gpu_handle(
    gpu_identifier: str,
    pynvml: Any,
) -> Any:
    if gpu_identifier.isdigit():
        return pynvml.nvmlDeviceGetHandleByIndex(int(gpu_identifier))
    return pynvml.nvmlDeviceGetHandleByUUID(gpu_identifier)


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
            "tracking_run_id": str(tracking_run_id),
        },
    )


def _read_tracking_metadata(
    config: DictConfig,
) -> Dict[str, Any]:
    path = _get_tracking_metadata_path(config=config)
    if not os.path.isfile(path):
        return {}
    with open(
        path,
        encoding="utf-8",
    ) as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid tracking metadata payload: {path}")
    return payload


def _validate_tracking_metadata_backend(
    config: DictConfig,
    metadata: Dict[str, Any],
) -> None:
    if len(metadata) == 0:
        return
    if metadata.get("backend") != str(config.tracking.backend):
        raise ValueError(
            "tracking_metadata.json backend does not match tracking.backend."
        )


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
    temp_path = f"{path}.tmp.{os.getpid()}"
    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")
    os.replace(
        temp_path,
        path,
    )


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
