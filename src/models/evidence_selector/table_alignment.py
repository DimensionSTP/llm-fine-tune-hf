from typing import Dict, List, Any

import torch

from scipy.optimize import linear_sum_assignment


def associate_table_records(
    scores: Dict[str, torch.Tensor],
    targets: Dict[str, Any],
    loss_weights: Dict[str, float],
) -> List[int]:
    cell_logits = scores["cell_logits"]
    records = targets["record_cell_positions"]
    request_count, field_count, class_count = cell_logits.shape
    if len(records) > request_count:
        raise ValueError(
            f"{len(records)} GT records cannot fit {request_count} input-derived candidates. "
            "Do not drop records or create input queries from GT."
        )
    if not records:
        return []
    if any(len(record) != field_count for record in records):
        raise ValueError("Record width differs from the supplied schema.")
    if any(
        type(value) is not int or not -1 <= value < class_count - 1
        for record in records
        for value in record
    ):
        raise ValueError("Record cells must be input positions or the valid -1 class.")
    with torch.no_grad():
        log_probabilities = cell_logits.detach().float().log_softmax(dim=-1)
        labels = torch.tensor(
            [
                [class_count - 1 if value == -1 else value for value in record]
                for record in records
            ],
            device=cell_logits.device,
            dtype=torch.long,
        )
        cost = cell_logits.new_zeros(
            (len(records), request_count),
            dtype=torch.float32,
        )
        for field_index in range(field_count):
            cost -= (
                log_probabilities[:, field_index, labels[:, field_index]].transpose(
                    0,
                    1,
                )
                * loss_weights["cell"]
                / field_count
            )
        record_logits = scores["record_logits"].detach().float()
        if record_logits.shape != (request_count,):
            raise ValueError("Result eligibility must match input candidate count.")
        cost -= (
            loss_weights["record"]
            * len(records)
            / request_count
            * record_logits.unsqueeze(0)
        )
        record_positions, request_positions = linear_sum_assignment(cost.cpu().numpy())
    if record_positions.tolist() != list(range(len(records))):
        raise ValueError("Loss association must preserve every source record.")
    return request_positions.tolist()
