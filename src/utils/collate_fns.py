from typing import Dict, List

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
