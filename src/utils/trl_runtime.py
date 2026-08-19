from typing import Dict, List, Iterator, Any
from contextlib import contextmanager
from types import MethodType

from omegaconf import DictConfig

import torch

from transformers import PreTrainedModel, Trainer, TrainerCallback


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
    if _patch_sdpo_peft_ema(
        trainer=trainer,
        config=config,
    ):
        applied_patches.append("SDPO ZeRO-3 PEFT EMA synchronization")
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


def _patch_sdpo_peft_ema(
    trainer: Trainer,
    config: DictConfig,
) -> bool:
    if config.fine_tune_method != "sdpo":
        return False
    if not _is_zero3_enabled(trainer=trainer):
        return False

    callback = _find_peft_ema_callback(trainer=trainer)
    if callback is None:
        return False
    if hasattr(callback, "_trl_runtime_initialize_teacher_adapter"):
        return False

    callback._trl_runtime_initialize_teacher_adapter = (
        callback._initialize_teacher_adapter
    )
    callback._trl_runtime_get_student_state_dict = callback._get_student_state_dict
    callback._initialize_teacher_adapter = MethodType(
        _initialize_sdpo_teacher_adapter,
        callback,
    )
    callback._get_student_state_dict = MethodType(
        _get_sdpo_student_state_dict,
        callback,
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


def _find_peft_ema_callback(
    trainer: Trainer,
) -> TrainerCallback | None:
    from trl.experimental.sdpo.teacher_sync import PEFTAdapterEMACallback

    return next(
        (
            callback
            for callback in trainer.callback_handler.callbacks
            if isinstance(callback, PEFTAdapterEMACallback)
        ),
        None,
    )


def _initialize_sdpo_teacher_adapter(
    callback: TrainerCallback,
) -> None:
    with _gather_sdpo_student_parameters(callback=callback):
        callback._trl_runtime_initialize_teacher_adapter()


def _get_sdpo_student_state_dict(
    callback: TrainerCallback,
) -> Dict[str, torch.Tensor]:
    with _gather_sdpo_student_parameters(callback=callback):
        state_dict = callback._trl_runtime_get_student_state_dict()
        return {key: value.detach().clone() for key, value in state_dict.items()}


@contextmanager
def _gather_sdpo_student_parameters(
    callback: TrainerCallback,
) -> Iterator[None]:
    from deepspeed import zero

    model = callback.accelerator.unwrap_model(callback.model)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    with zero.GatheredParameters(
        trainable_parameters,
        modifier_rank=None,
        enabled=(
            _should_gather_parameters(
                callback=callback,
                parameters=trainable_parameters,
            )
        ),
    ):
        yield


def _should_gather_parameters(
    callback: TrainerCallback,
    parameters: List[torch.nn.Parameter],
) -> bool:
    deepspeed_plugin = callback.accelerator.state.deepspeed_plugin
    return (
        deepspeed_plugin is not None
        and deepspeed_plugin.zero_stage == 3
        and any(parameter.numel() == 0 for parameter in parameters)
    )
