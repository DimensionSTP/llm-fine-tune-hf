from typing import Dict, List, Optional, Any

import torch


def collate_fn_vlm(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        items = [item[key] for item in batch]
        if key == "pixel_values":
            collated[key] = torch.cat(
                items,
                dim=0,
            )
        elif key == "image_grid_thw":
            collated[key] = torch.cat(
                items,
                dim=0,
            )
        elif key in {
            "input_ids",
            "attention_mask",
            "pixel_position_ids",
        }:
            collated[key] = torch.stack(
                items,
                dim=0,
            )
        else:
            try:
                collated[key] = torch.stack(
                    items,
                    dim=0,
                )
            except:
                collated[key] = items
    return collated


class SFTDynamicPaddingCollator:
    def __init__(
        self,
        pad_token_id: int,
        pad_to_multiple_of: Optional[int] = None,
        ignore_index: int = -100,
    ) -> None:
        if pad_to_multiple_of is not None and pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be a positive integer or None.")
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.ignore_index = ignore_index

    def __call__(
        self,
        batch: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        self._validate_batch(batch=batch)
        output = {}
        for key in batch[0].keys():
            items = [sample[key] for sample in batch]
            if self._is_sequence_key(key=key):
                output[key] = self._pad_sequence_tensors(
                    key=key,
                    tensors=items,
                )
            elif key in ["pixel_position_ids", "pixel_attention_mask"]:
                output[key] = self._pad_vlm_first_dim(
                    key=key,
                    tensors=items,
                    padding_value=self._get_vlm_first_dim_padding_value(key=key),
                )
            elif key == "pixel_values" and self._uses_vlm_first_dim_padding(
                keys=list(batch[0].keys()),
            ):
                output[key] = self._pad_vlm_first_dim(
                    key=key,
                    tensors=items,
                    padding_value=0,
                )
            elif self._is_vlm_concat_key(key=key):
                output[key] = self._collate_vlm_tensors(
                    key=key,
                    tensors=items,
                )
            else:
                raise ValueError(f"Unsupported SFT dynamic collator key: {key}.")
        return output

    def _validate_batch(
        self,
        batch: List[Dict[str, Any]],
    ) -> None:
        if not batch:
            raise ValueError("SFT dynamic collator received an empty batch.")

        keys = set(batch[0].keys())
        for sample in batch:
            if set(sample.keys()) != keys:
                raise ValueError(
                    "All SFT dynamic collator samples must have the same keys."
                )
            for key, value in sample.items():
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"SFT dynamic collator only supports tensor values, got {type(value)} for key {key}."
                    )

    def _is_sequence_key(
        self,
        key: str,
    ) -> bool:
        return key in [
            "input_ids",
            "attention_mask",
            "labels",
            "token_type_ids",
            "mm_token_type_ids",
        ]

    def _is_vlm_concat_key(
        self,
        key: str,
    ) -> bool:
        return key in [
            "pixel_values",
            "image_grid_thw",
            "image_sizes",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
        ]

    def _get_padding_value(
        self,
        key: str,
    ) -> int:
        if key == "input_ids":
            return self.pad_token_id
        if key == "labels":
            return self.ignore_index
        return 0

    def _get_target_length(
        self,
        tensors: List[torch.Tensor],
    ) -> int:
        target_length = max(tensor.size(0) for tensor in tensors)
        if self.pad_to_multiple_of is not None:
            remainder = target_length % self.pad_to_multiple_of
            if remainder != 0:
                target_length += self.pad_to_multiple_of - remainder
        return target_length

    def _pad_sequence_tensors(
        self,
        key: str,
        tensors: List[torch.Tensor],
    ) -> torch.Tensor:
        for tensor in tensors:
            if tensor.dim() != 1:
                raise ValueError(
                    f"SFT dynamic sequence key {key} must be 1D, got shape {tuple(tensor.shape)}."
                )

        target_length = self._get_target_length(tensors=tensors)
        output = torch.full(
            (len(tensors), target_length),
            self._get_padding_value(key=key),
            dtype=tensors[0].dtype,
            device=tensors[0].device,
        )
        for idx, tensor in enumerate(tensors):
            output[idx, : tensor.size(0)] = tensor
        return output

    def _uses_vlm_first_dim_padding(
        self,
        keys: List[str],
    ) -> bool:
        return "image_grid_thw" not in keys and (
            "pixel_position_ids" in keys or "pixel_attention_mask" in keys
        )

    def _get_vlm_first_dim_padding_value(
        self,
        key: str,
    ) -> int:
        if key == "pixel_position_ids":
            return -1
        return 0

    def _pad_vlm_first_dim(
        self,
        key: str,
        tensors: List[torch.Tensor],
        padding_value: int,
    ) -> torch.Tensor:
        trailing_shape = tensors[0].shape[1:]
        for tensor in tensors:
            if tensor.dim() < 1:
                raise ValueError(
                    f"SFT dynamic VLM key {key} must have at least 1 dimension, got shape {tuple(tensor.shape)}."
                )
            if tensor.shape[1:] != trailing_shape:
                raise ValueError(
                    f"SFT dynamic VLM key {key} has incompatible trailing shapes."
                )

        target_length = max(tensor.size(0) for tensor in tensors)
        output = torch.full(
            (len(tensors), target_length, *trailing_shape),
            padding_value,
            dtype=tensors[0].dtype,
            device=tensors[0].device,
        )
        for idx, tensor in enumerate(tensors):
            output[idx, : tensor.size(0)] = tensor
        return output

    def _collate_vlm_tensors(
        self,
        key: str,
        tensors: List[torch.Tensor],
    ) -> torch.Tensor:
        if key in ["image_grid_thw", "image_sizes", "video_grid_thw"] and all(
            tensor.dim() == 1 for tensor in tensors
        ):
            return torch.stack(
                tensors,
                dim=0,
            )
        return torch.cat(
            tensors,
            dim=0,
        )
