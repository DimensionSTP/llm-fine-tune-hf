from typing import Dict, Any
import json
import logging
from urllib.request import Request, urlopen

from omegaconf import DictConfig

from .tracking import get_tracking_context
from .metadata_security import redact_metadata_text


def validate_notifications_config(
    config: DictConfig,
) -> None:
    backend = config.notifications.backend
    if backend not in {"disabled", "slack", "wandb"}:
        raise ValueError(f"Unsupported notifications backend: {backend}.")
    if backend == "slack" and (
        config.notifications.webhook_url is None
        or str(config.notifications.webhook_url).strip() == ""
    ):
        raise ValueError(
            "notifications.webhook_url is required when notifications.backend=slack."
        )
    if backend == "wandb" and config.tracking.backend != "wandb":
        raise ValueError("notifications.backend=wandb requires tracking.backend=wandb.")


def send_notification(
    config: DictConfig,
    title: str,
    text: str,
    level: str,
    status: str,
) -> None:
    validate_notifications_config(config=config)
    backend = config.notifications.backend
    if backend == "disabled":
        return
    tracking_context = get_tracking_context(config=config)
    message = _build_notification_message(
        config=config,
        tracking_context=tracking_context,
        title=title,
        text=text,
        level=level,
        status=status,
    )
    if backend == "slack":
        _send_slack_notification(
            config=config,
            message=message,
        )
        return
    if backend == "wandb":
        _send_wandb_notification(
            title=title,
            text=message,
            level=level,
        )
        return
    raise ValueError(f"Unsupported notifications backend: {backend}.")


def send_notification_preserving_error(
    config: DictConfig,
    title: str,
    text: str,
    level: str,
    status: str,
) -> None:
    try:
        send_notification(
            config=config,
            title=title,
            text=text,
            level=level,
            status=status,
        )
    except Exception as error:
        redacted_error = redact_metadata_text(
            config=config,
            text=str(error),
        )
        logging.getLogger(__name__).error(
            "Failed to send notification while preserving the pipeline result: "
            f"{redacted_error}"
        )


def _build_notification_message(
    config: DictConfig,
    tracking_context: Dict[str, Any],
    title: str,
    text: str,
    level: str,
    status: str,
) -> str:
    lines = [
        f"[{level}] {title}",
        text,
        f"status: {status}",
        f"project: {config.project_name}",
        f"run: {_resolve_notification_run_name(config=config)}",
        f"artifact_run_id: {config.run_id}",
        f"tracking_backend: {tracking_context['backend']}",
        f"tracking_run_id: {tracking_context['tracking_run_id']}",
        f"method: {config.fine_tune_method}",
        f"model: {config.model_detail}",
        f"dataset: {config.dataset_name}",
        f"output_dir: {config.output_dir}",
    ]
    if tracking_context["run_url"] is not None:
        lines.append(f"run_url: {tracking_context['run_url']}")
    return "\n".join(lines)


def _resolve_notification_run_name(
    config: DictConfig,
) -> str:
    if config.mode == "train":
        return str(config.logging_name)
    return str(config.model_detail)


def _send_slack_notification(
    config: DictConfig,
    message: str,
) -> None:
    request = Request(
        url=str(config.notifications.webhook_url),
        data=json.dumps({"text": message}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(
        request,
        timeout=config.notifications.timeout_seconds,
    ) as response:
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(
                f"Slack notification failed with HTTP status {response.status}."
            )


def _send_wandb_notification(
    title: str,
    text: str,
    level: str,
) -> None:
    try:
        import wandb
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "notifications.backend=wandb requires wandb. Install project dependencies first."
        ) from error
    if wandb.run is None:
        raise RuntimeError("W&B notification requires an active W&B run.")
    wandb.run.alert(
        title=title,
        text=text,
        level=level,
    )
