from typing import Dict, List, Any
from contextlib import ExitStack

import torch

from transformers import ProcessorMixin

from PIL import Image

from .selection_lengths import ModelLengths


class SelectionCollator:
    def __init__(
        self,
        processor: ProcessorMixin,
        lengths: ModelLengths,
        model_name: str,
        selection_config: Dict[str, Any],
        max_length: int,
    ) -> None:
        self.processor = processor
        self.lengths = lengths
        self.model_name = model_name
        self.config = selection_config
        if type(max_length) is not int or max_length < 1:
            raise ValueError("Selection max_length must be a positive integer.")
        self.max_length = max_length

    def __call__(
        self,
        rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not rows:
            raise ValueError("A native batch cannot be empty.")
        features = [row["inputs"] for row in rows]
        input_lengths = [
            self.lengths.analyze(
                features=feature,
            )
            for feature in features
        ]
        if any(item["input_ids_length"] > self.max_length for item in input_lengths):
            raise ValueError(
                "Selection inputs exceed max_length; truncation is forbidden."
            )
        rendered_texts = [item["text"] for item in input_lengths]
        with ExitStack() as stack:
            images = [
                stack.enter_context(Image.open(path)).convert("RGB")
                for feature in features
                for path in feature["images"]
            ]
            model_inputs = dict(
                self.processor(
                    text=rendered_texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                ),
            )
        readouts = []
        for index, (feature, length_metadata) in enumerate(
            zip(
                features,
                input_lengths,
                strict=True,
            ),
        ):
            saved_metadata = feature["pooling_and_lengths"][self.model_name]
            for name in (
                "input_ids_length",
                "field_ends",
                "ocr_ends",
                "cell_ends",
                "image_grids",
            ):
                if saved_metadata[name] != length_metadata[name]:
                    raise ValueError(f"Stale native readout/length metadata: {name}")
            attention_mask = model_inputs["attention_mask"][index]
            active_positions = attention_mask.nonzero(as_tuple=True)[0]
            if len(active_positions) != length_metadata["input_ids_length"]:
                raise ValueError(
                    "Native processor length differs from the validated token map."
                )
            padding_offset = int(active_positions[0])
            expected_positions = torch.arange(
                padding_offset,
                padding_offset + len(active_positions),
                device=active_positions.device,
            )
            if not torch.equal(
                active_positions,
                expected_positions,
            ):
                raise ValueError(
                    "Only contiguous left/right sequence padding is supported."
                )
            candidate_key = "ocr_ends" if feature["task"] == "kv" else "cell_ends"
            if feature["task"] == "table":
                for key in ("row_request_contract", "record_request_contract"):
                    if feature[key] != self.config["table"][key]:
                        raise ValueError(
                            f"Table input contract differs from the head config: {key}"
                        )
            readouts.append(
                {
                    "task": feature["task"],
                    "field_ends": [
                        position + padding_offset
                        for position in length_metadata["field_ends"]
                    ],
                    "candidate_ends": [
                        position + padding_offset
                        for position in length_metadata[candidate_key]
                    ],
                    "row_requests": feature["row_requests"],
                    "record_requests": feature["record_requests"]
                    if feature["task"] == "table"
                    else [],
                },
            )
        actual_grids = model_inputs["image_grid_thw"].tolist()
        expected_grids = [
            image["grid_thw"] for item in input_lengths for image in item["image_grids"]
        ]
        if actual_grids != expected_grids:
            raise ValueError(
                "Native image grids differ from the validated input contract."
            )
        return {
            "model_inputs": model_inputs,
            "readouts": readouts,
            "targets": [row["labels"] for row in rows],
            "input_ids": [row["input_id"] for row in rows],
        }
