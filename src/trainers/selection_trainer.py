from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint

from peft import load_peft_weights, set_peft_model_state_dict

from ..models.evidence_selector import (
    VLMSelector,
    save_selection_checkpoint,
    validate_selection_checkpoint,
    load_selection_weights,
)


class SelectionTrainer(Trainer):
    def __init__(
        self,
        input_contract: Dict[str, Any],
        model_reference: Dict[str, Any],
        head_learning_rate: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        arguments = kwargs["args"]
        if arguments.n_gpu > 1:
            raise ValueError(
                "Launch selection with Accelerate: use one process per GPU."
            )
        if arguments.fsdp:
            raise ValueError("Use DDP or DeepSpeed for distributed selection.")
        if arguments.use_liger_kernel:
            raise ValueError("Non-generative selection does not use Liger kernels.")
        if (
            arguments.dataloader_num_workers != 0
            or arguments.dataloader_persistent_workers
            or arguments.dataloader_prefetch_factor is not None
        ):
            raise ValueError("Selection requires workers=0 without persistent workers.")
        if arguments.remove_unused_columns:
            raise ValueError("Selection needs its nested input and target columns.")
        if arguments.load_best_model_at_end:
            raise ValueError(
                "Best-model selection is not part of this initial integration."
            )
        self.input_contract = input_contract
        self.model_reference = model_reference
        self.head_learning_rate = (
            arguments.learning_rate
            if head_learning_rate is None
            else head_learning_rate
        )
        super().__init__(**kwargs)
        self.loss_is_scaled_for_ga = True
        self.model_accepts_loss_kwargs = False
        self.label_names = ["targets"]

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        **kwargs: Any,
    ) -> Any:
        if resume_from_checkpoint and self.is_deepspeed_enabled:
            if resume_from_checkpoint is True:
                resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
                if resume_from_checkpoint is None:
                    raise ValueError(
                        f"No valid checkpoint found in output directory ({self.args.output_dir})"
                    )
            model = self.accelerator.unwrap_model(self.model)
            validate_selection_checkpoint(
                selector=model.selector,
                checkpoint=Path(resume_from_checkpoint) / "selection_model",
                selection_config=model.selection_config,
                input_contract=self.input_contract,
                model_reference=self.model_reference,
            )
        return super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            **kwargs,
        )

    def create_optimizer(
        self,
        model: Optional[torch.nn.Module] = None,
    ) -> torch.optim.Optimizer:
        if self.optimizer is not None:
            return self.optimizer
        model = self.model if model is None else model
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        decay_names = set(self.get_decay_parameter_names(model))
        groups = [
            {
                "params": [
                    parameter
                    for name, parameter in parameters.items()
                    if name.startswith("selector.") == is_head
                    and (name in decay_names) == decay
                ],
                "lr": self.head_learning_rate if is_head else self.args.learning_rate,
                "weight_decay": self.args.weight_decay if decay else 0.0,
            }
            for is_head in (False, True)
            for decay in (True, False)
        ]
        optimizer_class, optimizer_kwargs = (
            self.optimizer_cls_and_kwargs
            if self.optimizer_cls_and_kwargs is not None
            else self.get_optimizer_cls_and_kwargs(
                args=self.args,
                model=model,
            )
        )
        self.optimizer = optimizer_class(
            [group for group in groups if group["params"]],
            **optimizer_kwargs,
        )
        return self.optimizer

    def get_eval_dataloader(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> DataLoader:
        even_batches = self.accelerator.even_batches
        self.accelerator.even_batches = True
        try:
            return super().get_eval_dataloader(
                *args,
                **kwargs,
            )
        finally:
            self.accelerator.even_batches = even_batches

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        scores = model(
            model_inputs=inputs["model_inputs"],
            readouts=inputs["readouts"],
        )
        selector = self.accelerator.unwrap_model(model)
        losses = selector.compute_loss(
            scores=scores,
            targets=inputs["targets"],
        )["per_example_losses"]
        loss = (
            losses.mean()
            if num_items_in_batch is None
            else losses.sum() / num_items_in_batch
        )
        if num_items_in_batch is not None and self.args.average_tokens_across_devices:
            loss *= self.accelerator.num_processes // self.get_tp_size()
        if return_outputs:
            return (
                loss,
                {"scores": scores},
            )
        return loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, None, None]:
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(
                model=model,
                inputs=inputs,
            )
        return (
            loss.detach(),
            None,
            None,
        )

    def save_model(
        self,
        output_dir: Optional[str] = None,
        _internal_call: bool = False,
    ) -> None:
        if self.is_deepspeed_enabled and self.deepspeed.zero_optimization_stage() == 3:
            state_dict = self.deepspeed._zero3_consolidated_16bit_state_dict(
                exclude_frozen_parameters=True,
            )
            if self.args.should_save:
                self._save(
                    output_dir=output_dir,
                    state_dict=state_dict,
                )
            return
        super().save_model(
            output_dir=output_dir,
            _internal_call=_internal_call,
        )

    def _get_num_items_in_batch(
        self,
        batch_samples: List[Dict[str, Any]],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not batch_samples:
            return None
        count = torch.tensor(
            sum(len(batch["readouts"]) for batch in batch_samples),
            device=device,
        )
        if self.args.average_tokens_across_devices and self.args.world_size > 1:
            count = self.accelerator.gather(count).sum()
        count = count // (self.get_tp_size() * self.get_cp_size() * self.get_sp_size())
        return count

    def _save(
        self,
        output_dir: Optional[str] = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        model = self.accelerator.unwrap_model(self.model)
        if not isinstance(model, VLMSelector) or not model.peft_wrapped:
            raise ValueError("Selection checkpoints require the VLM's trainable LoRA.")
        adapter_state_dict = None
        selector_state_dict = None
        if state_dict is not None:
            adapter_state_dict = {
                name.removeprefix("vlm."): value
                for name, value in state_dict.items()
                if name.startswith("vlm.")
            }
            selector_state_dict = {
                name.removeprefix("selector."): value
                for name, value in state_dict.items()
                if name.startswith("selector.")
            }
            expected_head_keys = set(model.selector.state_dict())
            if not adapter_state_dict or set(selector_state_dict) != expected_head_keys:
                raise ValueError(
                    "Gathered selection state must contain LoRA and all heads."
                )
        output = Path(self.args.output_dir if output_dir is None else output_dir)
        output.mkdir(
            parents=True,
            exist_ok=True,
        )
        save_selection_checkpoint(
            adapter=model.vlm,
            selector=model.selector,
            output=output / "selection_model",
            selection_config=model.selection_config,
            input_contract=self.input_contract,
            model_reference=self.model_reference,
            adapter_state_dict=adapter_state_dict,
            selector_state_dict=selector_state_dict,
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output)
        torch.save(
            self.args,
            output / "training_args.bin",
        )

    def _load_from_checkpoint(
        self,
        resume_from_checkpoint: str,
        model: Optional[torch.nn.Module] = None,
    ) -> None:
        selector_model = self.accelerator.unwrap_model(
            self.model if model is None else model,
        )
        checkpoint_dir = Path(resume_from_checkpoint) / "selection_model"
        load_selection_weights(
            selector=selector_model.selector,
            checkpoint=checkpoint_dir,
            selection_config=selector_model.selection_config,
            input_contract=self.input_contract,
            model_reference=self.model_reference,
        )
        weights = load_peft_weights(
            model_id=str(checkpoint_dir),
            device="cpu",
            local_files_only=True,
        )
        set_peft_model_state_dict(
            model=selector_model.vlm,
            peft_model_state_dict=weights,
            adapter_name="default",
        )
