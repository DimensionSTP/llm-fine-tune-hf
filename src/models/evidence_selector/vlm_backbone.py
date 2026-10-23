from typing import Dict, List, Optional, Any

import torch
from torch import nn

from .kv_selector import KVSelector
from .table_selector import TableSelector
from .selection_heads import pool_endpoints
from .table_alignment import associate_table_records


class VLMSelector(nn.Module):
    def __init__(
        self,
        vlm: nn.Module,
        task: str,
        config: Dict[str, Any],
        peft_wrapped: bool,
    ) -> None:
        super().__init__()
        if config["pooling"] != "record_end" or config["confidence_enabled"]:
            raise ValueError(
                "This implementation uses endpoint pooling without confidence."
            )
        if task not in ("kv", "table"):
            raise ValueError("The selector task must be kv or table.")
        self.vlm = vlm
        self.task = task
        self.selection_config = config
        self.peft_wrapped = peft_wrapped
        base_model = self.base_model()
        self.config = base_model.config
        if base_model.config.model_type not in ("qwen3_5", "qwen3_vl"):
            raise ValueError(
                "The native model family needs an explicitly verified path."
            )
        native_model = base_model.model
        if not isinstance(
            native_model.visual,
            nn.Module,
        ) or not isinstance(native_model.language_model, nn.Module):
            raise ValueError("A complete native multimodal model is required.")
        hidden_size = base_model.config.text_config.hidden_size
        selector_class = KVSelector if task == "kv" else TableSelector
        self.selector = selector_class(
            hidden_size=hidden_size,
            config=config,
        )

    def forward(
        self,
        model_inputs: Dict[str, torch.Tensor],
        readouts: List[Dict[str, Any]],
    ) -> List[Dict[str, torch.Tensor]]:
        allowed_keys = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "pixel_values",
            "image_grid_thw",
            "mm_token_type_ids",
        }
        if set(model_inputs) - allowed_keys:
            raise ValueError(
                "Non-native inputs or supervision reached the VLM forward."
            )
        if len(readouts) != model_inputs["input_ids"].shape[0]:
            raise ValueError("Readout count differs from the native input batch.")
        outputs = self.base_model().model(
            **model_inputs,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        predictions = []
        for index, readout in enumerate(readouts):
            if set(readout) != {
                "task",
                "field_ends",
                "candidate_ends",
                "row_requests",
                "record_requests",
            }:
                raise ValueError(
                    "Only input-derived positions and row requests belong in readouts."
                )
            if readout["task"] != self.task:
                raise ValueError("One selector cannot mix KV and Table tasks.")
            valid_positions = readout["field_ends"] + readout["candidate_ends"]
            attention_mask = model_inputs["attention_mask"][index]
            if any(
                not 0 <= position < len(attention_mask) or not attention_mask[position]
                for position in valid_positions
            ):
                raise ValueError("A field or candidate readout points into padding.")
            field_states = pool_endpoints(
                hidden_states=hidden_states[index],
                positions=readout["field_ends"],
            )
            candidate_states = pool_endpoints(
                hidden_states=hidden_states[index],
                positions=readout["candidate_ends"],
            )
            selector_dtype = next(self.selector.parameters()).dtype
            selector_inputs = {
                "field_states": field_states.to(dtype=selector_dtype),
                "candidate_states": candidate_states.to(dtype=selector_dtype),
            }
            if self.task == "table":
                selector_inputs["row_requests"] = readout["row_requests"]
                selector_inputs["record_requests"] = readout["record_requests"]
            predictions.append(self.selector(**selector_inputs))
        return predictions

    def get_input_embeddings(self) -> nn.Module:
        return self.vlm.get_input_embeddings()

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None,
        every_n_layers: int = 1,
        offload: bool = False,
    ) -> None:
        self.vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            every_n_layers=every_n_layers,
            offload=offload,
        )

    def gradient_checkpointing_disable(self) -> None:
        self.vlm.gradient_checkpointing_disable()

    def base_model(self) -> nn.Module:
        return self.vlm.get_base_model() if self.peft_wrapped else self.vlm

    def compute_loss(
        self,
        scores: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not scores or len(scores) != len(targets):
            raise ValueError(
                "Scores and supervision must contain the same nonempty batch."
            )
        losses = []
        record_associations = []
        for prediction, target in zip(
            scores,
            targets,
            strict=True,
        ):
            if self.task == "kv":
                loss_outputs = self.selector.compute_loss(
                    scores=prediction,
                    targets=target,
                )
            else:
                record_request_positions = associate_table_records(
                    scores=prediction,
                    targets=target,
                    loss_weights=self.selection_config["table"]["loss_weights"],
                )
                record_associations.append(record_request_positions)
                loss_outputs = self.selector.compute_loss(
                    scores=prediction,
                    targets=target,
                    record_request_positions=record_request_positions,
                )
            losses.append(loss_outputs["loss"])
        per_example_losses = torch.stack(losses)
        return {
            "loss": per_example_losses.mean(),
            "per_example_losses": per_example_losses,
            "record_request_positions_loss_only": record_associations,
        }

    def predict(
        self,
        scores: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, Any]]:
        return [self.selector.predict(scores=score) for score in scores]
