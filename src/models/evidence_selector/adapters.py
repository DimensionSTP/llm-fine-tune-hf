from typing import Dict, List, Any

from .kv_selector import validate_field_target


def restore_prediction(
    task: str,
    field_paths: List[List[str]],
    restoration: Dict[str, Any],
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    if task == "table":
        cells = restoration["cell_ids"]
        records = prediction["record_cell_positions"]
        if any(
            type(position) is not int or not -1 <= position < len(cells)
            for record in records
            for position in record
        ):
            raise ValueError("A predicted cell must be an input position or -1.")
        return {
            "c": [
                [cells[position] if position != -1 else -1 for position in record]
                for record in records
            ],
        }
    if task != "kv":
        raise ValueError("The selector task must be kv or table.")
    if any(len(path) < 2 or path[0] != "schema" for path in field_paths):
        raise ValueError("KV readout paths must start at the native user schema.")
    fields = prediction["field_targets"]
    if len(field_paths) != len(fields):
        raise ValueError("Predicted fields must match the input schema.")
    output = {}
    for path, target in zip(
        field_paths,
        fields,
        strict=True,
    ):
        validate_field_target(
            target=target,
            candidate_count=len(restoration["ocr_ids"]),
        )
        selected = target["selected_positions"]
        groups = []
        start = 0
        for end in target["group_end_offsets"]:
            groups.append(
                [restoration["ocr_ids"][position] for position in selected[start:end]]
            )
            start = end
        current = output
        for name in path[1:-1]:
            current = current.setdefault(
                name,
                {},
            )
        current[path[-1]] = groups
    return output
