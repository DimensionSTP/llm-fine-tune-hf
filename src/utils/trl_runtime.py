from typing import List, Any
from types import MethodType

from omegaconf import DictConfig

from transformers import PreTrainedModel, Trainer


def patch_trl_trainer_runtime(
    trainer: Trainer,
    config: DictConfig,
) -> List[str]:
    applied_patches: List[str] = []
    if _patch_gkd_gradient_checkpointing(
        trainer=trainer,
        config=config,
    ):
        applied_patches.append("GKD gradient checkpointing restoration")
    return applied_patches


def _patch_gkd_gradient_checkpointing(
    trainer: Trainer,
    config: DictConfig,
) -> bool:
    if config.fine_tune_method != "gkd":
        return False
    if not trainer.args.gradient_checkpointing:
        return False
    if not _is_zero3_enabled(trainer=trainer):
        return False

    model = trainer.accelerator.unwrap_model(trainer.model)
    if hasattr(model, "_trl_runtime_gradient_checkpointing_enable"):
        return False

    model._trl_runtime_gradient_checkpointing_enable = (
        model.gradient_checkpointing_enable
    )
    model._trl_runtime_gradient_checkpointing_kwargs = dict(
        trainer.args.gradient_checkpointing_kwargs
    )
    model.gradient_checkpointing_enable = MethodType(
        _enable_gradient_checkpointing,
        model,
    )
    return True


def _enable_gradient_checkpointing(
    model: PreTrainedModel,
    *args: Any,
    **kwargs: Any,
) -> None:
    if args or kwargs:
        model._trl_runtime_gradient_checkpointing_enable(
            *args,
            **kwargs,
        )
        return
    model._trl_runtime_gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=(
            model._trl_runtime_gradient_checkpointing_kwargs
        ),
    )


def _is_zero3_enabled(
    trainer: Trainer,
) -> bool:
    deepspeed_plugin = trainer.accelerator.state.deepspeed_plugin
    return deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
