from typing import Dict, List, Any

import torch
from torch import nn
import torch.nn.functional as F

from .selection_heads import SelectionProjection, BilinearScores


class TableSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        config: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.task = "table"
        policy = config["table"]
        if policy["loss_association"] != "post_forward_cell_nll_and_record_bce_delta":
            raise ValueError("Table association must remain a loss-only operation.")
        if (
            policy["request_mode"] != "input_row_and_cell_candidates"
            or policy["row_pooling"] != "mean_cell_vectors"
        ):
            raise ValueError("Table uses uniform input row/cell result candidates.")
        if policy["candidate_hard_mask"] or not policy["record_head_enabled"]:
            raise ValueError(
                "All cells and the approved result-eligibility head are required."
            )
        self.config = policy
        self.field_projection = SelectionProjection(
            hidden_size=hidden_size,
            config=config["projection"],
        )
        self.candidate_projection = SelectionProjection(
            hidden_size=hidden_size,
            config=config["projection"],
        )
        dimension = self.field_projection.output_size
        self.field_row_fusion = nn.Sequential(
            nn.Linear(
                2 * dimension,
                dimension,
            ),
            nn.GELU(),
            nn.Linear(
                dimension,
                dimension,
            ),
        )
        self.selection = BilinearScores(dimension=dimension)
        self.request_kind = nn.Embedding(
            2,
            dimension,
        )
        self.record_classifier = nn.Linear(
            dimension,
            1,
        )
        self.null_candidate = nn.Parameter(
            torch.zeros(
                1,
                dimension,
            )
        )
        nn.init.normal_(
            self.null_candidate,
            std=0.02,
        )

    def forward(
        self,
        field_states: torch.Tensor,
        candidate_states: torch.Tensor,
        row_requests: List[Dict[str, Any]],
        record_requests: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        fields = self.field_projection(hidden_states=field_states)
        cells = self.candidate_projection(hidden_states=candidate_states)
        row_states = []
        for request in row_requests:
            positions = request["cell_positions"]
            if not positions or any(
                type(position) is not int or not 0 <= position < len(cells)
                for position in positions
            ):
                raise ValueError("Every input row must reference valid input cells.")
            row_states.append(cells[positions].mean(dim=0))
        row_states = (
            torch.stack(row_states)
            if row_states
            else cells.new_empty((0, cells.shape[1]))
        )
        request_kind_states = self.request_kind(
            torch.arange(
                2,
                device=cells.device,
            ),
        )
        record_states = []
        for request in record_requests:
            row_position = request["row_request_position"]
            if type(row_position) is not int or not 0 <= row_position < len(row_states):
                raise ValueError(
                    "A result candidate must reference an input row context."
                )
            if request["kind"] == "row_context":
                if request["anchor_cell_position"] is not None:
                    raise ValueError("A row candidate cannot have a cell anchor.")
                record_states.append(
                    row_states[row_position] + request_kind_states[0],
                )
            elif request["kind"] == "cell_anchor":
                position = request["anchor_cell_position"]
                if (
                    type(position) is not int
                    or position not in row_requests[row_position]["cell_positions"]
                ):
                    raise ValueError(
                        "The cell anchor must belong to its input context."
                    )
                record_states.append(
                    (row_states[row_position] + cells[position]) / 2
                    + request_kind_states[1],
                )
            else:
                raise ValueError("Unknown input-derived result candidate kind.")
        record_states = (
            torch.stack(record_states)
            if record_states
            else cells.new_empty((0, cells.shape[1])) + request_kind_states.sum() * 0.0
        )
        row_fields = torch.cat(
            (
                fields.unsqueeze(0).expand(
                    len(record_states),
                    -1,
                    -1,
                ),
                record_states.unsqueeze(1).expand(
                    -1,
                    len(fields),
                    -1,
                ),
            ),
            dim=-1,
        )
        queries = self.field_row_fusion(row_fields)
        candidates = torch.cat(
            (cells, self.null_candidate),
            dim=0,
        )
        schema_context = self.field_row_fusion(
            torch.cat(
                (
                    fields.mean(dim=0).expand_as(record_states),
                    record_states,
                ),
                dim=-1,
            )
        )
        return {
            "cell_logits": self.selection(
                queries=queries,
                candidates=candidates,
            ),
            "record_logits": self.record_classifier(schema_context).squeeze(-1),
        }

    def compute_loss(
        self,
        scores: Dict[str, torch.Tensor],
        targets: Dict[str, Any],
        record_request_positions: List[int],
    ) -> Dict[str, torch.Tensor]:
        cell_logits = scores["cell_logits"].float()
        records = targets["record_cell_positions"]
        if len(record_request_positions) != len(records):
            raise ValueError(
                "Every source record needs a loss-only request association."
            )
        if len(record_request_positions) != len(set(record_request_positions)):
            raise ValueError(
                "Conflicting records cannot silently share one output request."
            )
        if any(
            type(position) is not int or not 0 <= position < len(cell_logits)
            for position in record_request_positions
        ):
            raise ValueError("A loss association is outside input requests.")
        null_index = cell_logits.shape[-1] - 1
        if any(len(record) != cell_logits.shape[1] for record in records):
            raise ValueError(
                "The supervised field count differs from the input schema."
            )
        if any(
            type(value) is not int or value < -1 or value >= null_index
            for record in records
            for value in record
        ):
            raise ValueError("A supervised cell is outside input candidates.")
        if not records:
            cell_loss = cell_logits.sum() * 0.0
        else:
            labels = torch.tensor(
                [
                    [null_index if value == -1 else value for value in record]
                    for record in records
                ],
                dtype=torch.long,
                device=cell_logits.device,
            )
            cell_loss = F.cross_entropy(
                cell_logits[record_request_positions].flatten(
                    0,
                    1,
                ),
                labels.flatten(),
            )
        record_logits = scores["record_logits"].float()
        if record_logits.shape != (len(cell_logits),):
            raise ValueError(
                "Each result candidate requires its own eligibility logit."
            )
        record_labels = torch.zeros_like(record_logits)
        record_labels[record_request_positions] = 1.0
        record_loss = (
            F.binary_cross_entropy_with_logits(
                record_logits,
                record_labels,
            )
            if record_logits.numel()
            else record_logits.sum() * 0.0
        )
        weights = self.config["loss_weights"]
        return {
            "loss": weights["cell"] * cell_loss + weights["record"] * record_loss,
            "cell": cell_loss,
            "record": record_loss,
        }

    def predict(
        self,
        scores: Dict[str, torch.Tensor],
    ) -> Dict[str, List[List[int]]]:
        cell_logits = scores["cell_logits"]
        null_index = cell_logits.shape[-1] - 1
        predictions = cell_logits.argmax(dim=-1).tolist()
        record_mask = (
            scores["record_logits"].sigmoid() >= self.config["record_threshold"]
        )
        return {
            "record_cell_positions": [
                [-1 if position == null_index else position for position in record]
                for index, record in enumerate(predictions)
                if record_mask[index]
            ],
        }
