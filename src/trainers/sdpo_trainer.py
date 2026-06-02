from typing import Dict, List, Tuple, Union, Optional, Any
from contextlib import AbstractContextManager

import importlib

datasets = importlib.import_module("datasets")
HFDataset = datasets.Dataset
HFIterableDataset = datasets.IterableDataset

import torch
from torch import nn

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.utils import is_peft_available

from trl.experimental.sdpo import SDPOConfig
from trl.experimental.sdpo import SDPOTrainer as TRLSDPOTrainer
from trl.trainer.utils import use_adapter

from accelerate.utils import GatheredParameters, is_peft_model

if is_peft_available():
    from peft.peft_model import PeftModel


class _ZeRO3PEFTAdapterEMACallback(TrainerCallback):
    def __init__(
        self,
        model: nn.Module,
        teacher_adapter_name: str,
        update_rate: float,
        sync_steps: int,
        accelerator: Any,
    ) -> None:
        self.model = model
        self.teacher_adapter_name = teacher_adapter_name
        self.update_rate = update_rate
        self.sync_steps = sync_steps
        self.accelerator = accelerator
        self.shadow_weights: Optional[Dict[str, torch.Tensor]] = None
        self._initialized = False

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._sync_accelerator(kwargs=kwargs)
        self._initialize_teacher_adapter()

    @torch.no_grad()
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if state.global_step % self.sync_steps != 0:
            return

        self._sync_accelerator(kwargs=kwargs)
        if not self._initialized:
            self._initialize_teacher_adapter()

        if self.shadow_weights is None:
            return

        model = self._get_unwrapped_peft_model()
        student_adapter_name = self._get_active_adapter_name(model=model)

        for key, student_param, teacher_param in self._iter_lora_parameter_pairs(
            model=model,
            student_adapter_name=student_adapter_name,
        ):
            self._update_teacher_parameter(
                key=key,
                student_param=student_param,
                teacher_param=teacher_param,
            )

    def _sync_accelerator(
        self,
        kwargs: Dict[str, Any],
    ) -> None:
        if self.accelerator is None and "accelerator" in kwargs:
            self.accelerator = kwargs["accelerator"]

    def _initialize_teacher_adapter(
        self,
    ) -> None:
        if self._initialized:
            return

        model = self._get_unwrapped_peft_model()
        student_adapter_name = self._get_active_adapter_name(model=model)
        teacher_adapter_config = model.peft_config[student_adapter_name]

        if self.teacher_adapter_name not in model.peft_config:
            model.add_adapter(
                adapter_name=self.teacher_adapter_name,
                peft_config=teacher_adapter_config,
            )

        shadow_weights: Dict[str, torch.Tensor] = {}
        for key, student_param, teacher_param in self._iter_lora_parameter_pairs(
            model=model,
            student_adapter_name=student_adapter_name,
        ):
            with self._gather_parameters(parameters=[student_param, teacher_param]):
                shadow_weights[key] = torch.zeros_like(
                    student_param.detach(),
                    device="cpu",
                )
                teacher_param.data.zero_()
                teacher_param.requires_grad_(False)

        model.set_adapter(student_adapter_name)
        self.shadow_weights = shadow_weights
        self._initialized = True

    def _get_unwrapped_peft_model(
        self,
    ) -> Any:
        if self.accelerator is None:
            model = self.model
        else:
            model = self.accelerator.unwrap_model(self.model)

        if not is_peft_available() or not isinstance(model, PeftModel):
            raise RuntimeError(
                "SDPO PEFT EMA teacher requires an unwrapped PeftModel student."
            )
        return model

    def _get_active_adapter_name(
        self,
        model: Any,
    ) -> str:
        active_adapter = model.active_adapter
        if active_adapter is None:
            return "default"
        if isinstance(active_adapter, str):
            return active_adapter
        if isinstance(active_adapter, list) and len(active_adapter) == 1:
            return active_adapter[0]
        raise RuntimeError(
            f"SDPO PEFT EMA teacher expects one active adapter, got {active_adapter}."
        )

    def _iter_lora_parameter_pairs(
        self,
        model: Any,
        student_adapter_name: str,
    ) -> List[Tuple[str, torch.nn.Parameter, torch.nn.Parameter]]:
        pairs = []
        for module_name, module in model.named_modules():
            if not self._is_lora_module(module=module):
                continue
            if student_adapter_name not in module.lora_A:
                continue
            if self.teacher_adapter_name not in module.lora_A:
                continue

            pairs.append(
                (
                    f"{module_name}.lora_A.weight",
                    module.lora_A[student_adapter_name].weight,
                    module.lora_A[self.teacher_adapter_name].weight,
                )
            )
            pairs.append(
                (
                    f"{module_name}.lora_B.weight",
                    module.lora_B[student_adapter_name].weight,
                    module.lora_B[self.teacher_adapter_name].weight,
                )
            )

        if len(pairs) == 0:
            raise RuntimeError(
                "SDPO PEFT EMA teacher could not find LoRA student/teacher parameter pairs."
            )
        return pairs

    def _is_lora_module(
        self,
        module: Any,
    ) -> bool:
        return (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and self.teacher_adapter_name in module.lora_A
            and self.teacher_adapter_name in module.lora_B
        )

    def _update_teacher_parameter(
        self,
        key: str,
        student_param: torch.nn.Parameter,
        teacher_param: torch.nn.Parameter,
    ) -> None:
        if self.shadow_weights is None:
            return

        with self._gather_parameters(parameters=[student_param, teacher_param]):
            student_value = student_param.detach()
            shadow = self.shadow_weights[key].to(
                device=student_value.device,
                dtype=student_value.dtype,
            )
            shadow.mul_(1.0 - self.update_rate)
            shadow.add_(student_value, alpha=self.update_rate)
            teacher_param.data.copy_(
                shadow.to(
                    device=teacher_param.device,
                    dtype=teacher_param.dtype,
                )
            )
            self.shadow_weights[key] = shadow.detach().cpu()

    def _gather_parameters(
        self,
        parameters: List[torch.nn.Parameter],
    ) -> AbstractContextManager:
        return GatheredParameters(
            parameters,
            modifier_rank=None,
            enabled=self._should_gather_parameters(parameters=parameters),
        )

    def _should_gather_parameters(
        self,
        parameters: List[torch.nn.Parameter],
    ) -> bool:
        return self._is_zero3_enabled() and any(
            hasattr(parameter, "ds_id") for parameter in parameters
        )

    def _is_zero3_enabled(
        self,
    ) -> bool:
        if self.accelerator is None:
            return False
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        return deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3


class SDPOTrainer(TRLSDPOTrainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel, nn.Module],
        reward_funcs: Optional[Union[Any, List[Any]]] = None,
        args: Optional[SDPOConfig] = None,
        train_dataset: Optional[Union[HFDataset, HFIterableDataset]] = None,
        eval_dataset: Optional[
            Union[
                HFDataset,
                HFIterableDataset,
                Dict[str, Union[HFDataset, HFIterableDataset]],
            ]
        ] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, ProcessorMixin]
        ] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, List[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Any, Any] = (None, None),
        peft_config: Any = None,
    ) -> None:
        self._use_peft_ema_teacher = self._should_use_peft_ema_teacher(
            model=model,
            args=args,
            peft_config=peft_config,
        )
        original_teacher_regularization = None

        if self._use_peft_ema_teacher and args is not None:
            original_teacher_regularization = args.teacher_regularization
            args.teacher_regularization = "none"

        try:
            super().__init__(
                model=model,
                reward_funcs=reward_funcs,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=processing_class,
                reward_processing_classes=reward_processing_classes,
                callbacks=callbacks,
                optimizers=optimizers,
                peft_config=peft_config,
            )
        finally:
            if self._use_peft_ema_teacher and args is not None:
                args.teacher_regularization = original_teacher_regularization

        if self._use_peft_ema_teacher:
            self.teacher_model = None
            ema_callback = _ZeRO3PEFTAdapterEMACallback(
                model=self.model,
                teacher_adapter_name="teacher",
                update_rate=self.args.teacher_update_rate,
                sync_steps=1,
                accelerator=self.accelerator,
            )
            ema_callback._initialize_teacher_adapter()
            self.add_callback(ema_callback)

    def _should_use_peft_ema_teacher(
        self,
        model: Union[str, PreTrainedModel, nn.Module],
        args: Optional[SDPOConfig],
        peft_config: Any,
    ) -> bool:
        if args is None:
            return False
        if args.teacher_regularization != "ema":
            return False
        if not is_peft_available():
            return False
        return peft_config is not None or is_peft_model(model)

    def _get_teacher_context_for_self_distillation(
        self,
        model: Any,
    ) -> Any:
        if not self._use_peft_ema_teacher:
            return super()._get_teacher_context_for_self_distillation(model)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if not isinstance(unwrapped_model, PeftModel):
            raise RuntimeError(
                "SDPO PEFT EMA teacher requires an unwrapped PeftModel student."
            )
        if "teacher" not in unwrapped_model.peft_config:
            raise RuntimeError(
                "SDPO PEFT EMA teacher adapter was not initialized before teacher forward."
            )
        return use_adapter(
            unwrapped_model,
            adapter_name="teacher",
        )
