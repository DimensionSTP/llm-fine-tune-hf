from typing import Dict, List, Any
import bisect
from functools import lru_cache
import json
import math
from pathlib import Path

from transformers import ProcessorMixin

from tokenizers import Tokenizer

from PIL import Image


class CachedTokenizer:
    def __init__(
        self,
        root: Path,
        cache_size: int,
    ) -> None:
        self.root = root
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        tokenizer_config = json.loads((root / "tokenizer.json").read_text())
        assert tokenizer_config["normalizer"] in (None, {"type": "NFC"})
        pre_tokenizers = tokenizer_config["pre_tokenizer"]["pretokenizers"]
        assert pre_tokenizers[0]["type"] == "Split"
        assert pre_tokenizers[1]["type"] == "ByteLevel"
        assert pre_tokenizers[1]["add_prefix_space"] is False
        assert "[^\\r\\n\\p{L}\\p{N}]?" in pre_tokenizers[0]["pattern"]["Regex"]
        assert all(
            "\n{" not in item["content"] for item in tokenizer_config["added_tokens"]
        )
        self.encode = lru_cache(maxsize=cache_size)(self._encode)

    def _encode(
        self,
        text: str,
    ) -> Dict[str, List[int]]:
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        offsets = encoded.offsets
        return {
            "ids": encoded.ids,
            "starts": [item[0] for item in offsets],
            "ends": [item[1] for item in offsets],
        }


class ModelLengths:
    def __init__(
        self,
        root: Path,
        backend: CachedTokenizer,
        processor: ProcessorMixin,
        config: Dict[str, Any],
    ) -> None:
        self.backend = backend
        self.processor = processor
        self.config = config
        self.image_config = json.loads((root / "preprocessor_config.json").read_text())
        self.model_config = json.loads((root / "config.json").read_text())
        self.image_token_id = self.model_config["image_token_id"]
        assert backend.tokenizer.token_to_id("<|image_pad|>") == self.image_token_id
        self.images: Dict[str, Dict[str, Any]] = {}

    def render(
        self,
        features: Dict[str, Any],
    ) -> str:
        return self.processor.apply_chat_template(
            conversation=features["messages"],
            tokenize=False,
            add_generation_prompt=self.config["add_generation_prompt"],
            enable_thinking=self.config["enable_thinking"],
            add_vision_id=self.config["add_vision_id"],
        )

    def image_grid(
        self,
        path: str,
    ) -> Dict[str, Any]:
        if path in self.images:
            return self.images[path]
        with Image.open(path) as image:
            width, height = image.size
        factor = self.image_config["patch_size"] * self.image_config["merge_size"]
        minimum_pixels = self.image_config["size"]["shortest_edge"]
        maximum_pixels = self.image_config["size"]["longest_edge"]
        assert (
            min(
                width,
                height,
            )
            > 0
            and max(
                width,
                height,
            )
            / min(
                width,
                height,
            )
            <= 200
        )
        resized_height = round(height / factor) * factor
        resized_width = round(width / factor) * factor
        if resized_height * resized_width > maximum_pixels:
            resize_ratio = math.sqrt(height * width / maximum_pixels)
            resized_height = max(
                factor,
                math.floor(height / resize_ratio / factor) * factor,
            )
            resized_width = max(
                factor,
                math.floor(width / resize_ratio / factor) * factor,
            )
        elif resized_height * resized_width < minimum_pixels:
            resize_ratio = math.sqrt(minimum_pixels / (height * width))
            resized_height = math.ceil(height * resize_ratio / factor) * factor
            resized_width = math.ceil(width * resize_ratio / factor) * factor
        patch_size = self.image_config["patch_size"]
        grid = [1, resized_height // patch_size, resized_width // patch_size]
        result = {
            "width": width,
            "height": height,
            "resized_width": resized_width,
            "resized_height": resized_height,
            "grid_thw": grid,
            "image_tokens": math.prod(grid) // self.image_config["merge_size"] ** 2,
        }
        self.images[path] = result
        return result

    def analyze(
        self,
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        rendered = self.render(features=features)
        prompt = features["messages"][-1]["content"][-1]["text"]
        assert rendered.count(prompt) == 1
        assert rendered.count("<|image_pad|>") == len(features["images"])
        prefix_start = rendered.index(prompt)
        characters = features["readout_characters"]
        segment_boundaries = [
            0,
            prefix_start + characters["document_start"],
            prefix_start + characters["user_start"],
            prefix_start + characters["readout_start"],
            len(rendered),
        ]
        assert all(
            rendered[position - 1 : position + 1] == "\n{"
            for position in segment_boundaries[1:-1]
        )
        segments = [
            self.backend.encode(rendered[left:right])
            for left, right in zip(
                segment_boundaries,
                segment_boundaries[1:],
            )
        ]
        images = [self.image_grid(path=path) for path in features["images"]]
        image_tokens = sum(item["image_tokens"] for item in images)
        image_token_offset = image_tokens - len(images)
        segment_lengths = [len(item["ids"]) for item in segments]
        offsets = [0]
        for size in segment_lengths:
            offsets.append(offsets[-1] + size)
        assert segments[0]["ids"].count(self.image_token_id) == len(images)
        assert not any(self.image_token_id in item["ids"] for item in segments[1:])

        result = {
            "text": rendered,
            "input_ids_length": offsets[-1] + image_token_offset,
            "image_grids": images,
            "field_ends": _token_endpoints(
                segment=segments[2],
                character_ends=characters["field_ends"],
                character_offset=prefix_start - segment_boundaries[2],
                token_offset=offsets[2] + image_token_offset,
            ),
            "ocr_ends": _token_endpoints(
                segment=segments[3],
                character_ends=characters["ocr_ends"],
                character_offset=prefix_start - segment_boundaries[3],
                token_offset=offsets[3] + image_token_offset,
            ),
            "cell_ends": _token_endpoints(
                segment=segments[3],
                character_ends=characters["cell_ends"],
                character_offset=prefix_start - segment_boundaries[3],
                token_offset=offsets[3] + image_token_offset,
            ),
        }
        for name in ("field_ends", "ocr_ends", "cell_ends"):
            assert all(
                0 <= value < result["input_ids_length"] for value in result[name]
            )
        return result


def _token_endpoints(
    segment: Dict[str, List[int]],
    character_ends: List[int],
    character_offset: int,
    token_offset: int,
) -> List[int]:
    local_character_ends = [character_offset + end for end in character_ends]
    positions = [
        bisect.bisect_left(
            segment["starts"],
            end,
        )
        - 1
        for end in local_character_ends
    ]
    if any(position < 0 for position in positions) or any(
        segment["ends"][position] < end
        for position, end in zip(
            positions,
            local_character_ends,
            strict=True,
        )
    ):
        raise ValueError("A readout endpoint is outside its token segment.")
    if len(positions) != len(set(positions)):
        raise ValueError("Distinct candidate endpoints overlap after tokenization.")
    return [token_offset + position for position in positions]
