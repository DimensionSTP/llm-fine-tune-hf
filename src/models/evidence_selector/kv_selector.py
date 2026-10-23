from typing import Dict, Any

import torch
from torch import nn
import torch.nn.functional as F

from .selection_heads import SelectionProjection, BilinearScores


class KVSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        config: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.task = "kv"
        if config["kv"]["order_mode"] != "learned_rank_preserve_source_order":
            raise ValueError(
                "KV must preserve the supervised OCR order through learned ranking."
            )
        self.config = config["kv"]
        self.field_projection = SelectionProjection(
            hidden_size=hidden_size,
            config=config["projection"],
        )
        self.candidate_projection = SelectionProjection(
            hidden_size=hidden_size,
            config=config["projection"],
        )
        dimension = self.field_projection.output_size
        self.selection = BilinearScores(dimension=dimension)
        self.end = BilinearScores(dimension=dimension)
        self.rank = BilinearScores(dimension=dimension)

    def forward(
        self,
        field_states: torch.Tensor,
        candidate_states: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        fields = self.field_projection(hidden_states=field_states)
        candidates = self.candidate_projection(hidden_states=candidate_states)
        return {
            "selection_logits": self.selection(
                queries=fields,
                candidates=candidates,
            ),
            "end_logits": self.end(
                queries=fields,
                candidates=candidates,
            ),
            "rank_scores": self.rank(
                queries=fields,
                candidates=candidates,
            ),
        }

    def compute_loss(
        self,
        scores: Dict[str, torch.Tensor],
        targets: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        selection_logits = scores["selection_logits"].float()
        end_logits = scores["end_logits"].float()
        rank_scores = scores["rank_scores"].float()
        field_targets = targets["field_targets"]
        if (
            selection_logits.ndim != 2
            or len(field_targets) != selection_logits.shape[0]
        ):
            raise ValueError("KV targets must match the input field count.")
        selected_labels = torch.zeros_like(selection_logits)
        end_labels = torch.zeros_like(end_logits)
        rank_losses = []
        for field_index, target in enumerate(field_targets):
            positions = target["selected_positions"]
            offsets = target["group_end_offsets"]
            validate_field_target(
                target=target,
                candidate_count=selection_logits.shape[1],
            )
            selected_labels[field_index, positions] = 1.0
            end_positions = [positions[offset - 1] for offset in offsets]
            end_labels[field_index, end_positions] = 1.0
            if len(positions) > 1:
                ordered_scores = rank_scores[field_index, positions]
                rank_losses.append(
                    F.relu(
                        self.config["rank_margin"]
                        + ordered_scores[:-1]
                        - ordered_scores[1:],
                    ).mean(),
                )
        zero_loss = (
            selection_logits.sum() * 0.0
            + end_logits.sum() * 0.0
            + rank_scores.sum() * 0.0
        )
        selection_loss = (
            F.binary_cross_entropy_with_logits(
                selection_logits,
                selected_labels,
            )
            if selection_logits.numel()
            else zero_loss
        )
        end_loss = (
            F.binary_cross_entropy_with_logits(
                end_logits[selected_labels.bool()],
                end_labels[selected_labels.bool()],
            )
            if selected_labels.any()
            else zero_loss
        )
        rank_loss = torch.stack(rank_losses).mean() if rank_losses else zero_loss
        weights = self.config["loss_weights"]
        return {
            "loss": weights["selection"] * selection_loss
            + weights["end"] * end_loss
            + weights["rank"] * rank_loss,
            "selection": selection_loss,
            "end": end_loss,
            "rank": rank_loss,
        }

    def predict(
        self,
        scores: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        selected = (
            scores["selection_logits"].sigmoid() >= self.config["selection_threshold"]
        )
        end_mask = scores["end_logits"].sigmoid() >= self.config["end_threshold"]
        output = []
        for field_index in range(selected.shape[0]):
            positions = selected[field_index].nonzero(as_tuple=True)[0]
            order = torch.argsort(
                scores["rank_scores"][field_index, positions],
                stable=True,
            )
            positions = positions[order].tolist()
            offsets = [
                index + 1
                for index, position in enumerate(positions)
                if end_mask[field_index, position]
            ]
            if positions and (not offsets or offsets[-1] != len(positions)):
                offsets.append(len(positions))
            output.append(
                {
                    "selected_positions": positions,
                    "group_end_offsets": offsets,
                },
            )
        return {"field_targets": output}


def validate_field_target(
    target: Dict[str, Any],
    candidate_count: int,
) -> None:
    positions = target["selected_positions"]
    offsets = target["group_end_offsets"]
    if len(positions) != len(set(positions)):
        raise ValueError("An OCR candidate cannot repeat within one field.")
    if any(
        type(position) is not int or not 0 <= position < candidate_count
        for position in positions
    ):
        raise ValueError("A selected OCR position is outside the input candidates.")
    if not positions:
        if offsets:
            raise ValueError("Empty selection cannot have a value-end offset.")
        return
    if not offsets or offsets[-1] != len(positions):
        raise ValueError("The final value must end at the final supervised OCR.")
    if any(type(offset) is not int for offset in offsets):
        raise ValueError("Value-end offsets must be integers.")
    if any(
        left >= right
        for left, right in zip(
            [0, *offsets[:-1]],
            offsets,
            strict=True,
        )
    ):
        raise ValueError("Value-end offsets must strictly increase.")
