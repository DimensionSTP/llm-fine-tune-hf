from typing import Dict, List, Optional, Any
import os

import importlib

datasets = importlib.import_module("datasets")
HFDataset = datasets.Dataset
load_dataset = datasets.load_dataset

import base64
import io
import math
import urllib.request

from PIL import Image


class StructuralDataset:
    def __init__(
        self,
        data_path: str,
        split_ratio: float,
        is_strict_split: bool,
        seed: int,
        dataset_name: str,
        dataset_format: str,
        data_column_name: str,
        solution_column_name: str,
        reward_categories_column_name: str,
        role_column_name: str,
        content_column_name: str,
        modality: str,
        max_pixels: Optional[int],
        do_resize: bool,
    ) -> None:
        self.data_path = data_path
        self.split_ratio = split_ratio
        self.is_strict_split = is_strict_split
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.data_column_name = data_column_name
        self.solution_column_name = solution_column_name
        self.reward_categories_column_name = reward_categories_column_name
        self.role_column_name = role_column_name
        self.content_column_name = content_column_name
        self._init_resize(
            modality=modality,
            max_pixels=max_pixels,
            do_resize=do_resize,
        )

    def __call__(self) -> Dict[str, HFDataset]:
        file_name = f"{self.dataset_name}.{self.dataset_format}"
        full_data_path = os.path.join(
            self.data_path,
            file_name,
        )

        dataset_format = self.dataset_format
        if dataset_format == "tsv":
            dataset_format = "csv"
        if dataset_format == "jsonl":
            dataset_format = "json"

        dataset = load_dataset(
            dataset_format,
            data_files=full_data_path,
        )["train"]

        output_column_names = [
            "prompt",
            self.solution_column_name,
            self.reward_categories_column_name,
        ]
        input_column_names = [
            self.data_column_name,
            self.solution_column_name,
            self.reward_categories_column_name,
        ]
        remove_columns = [
            name for name in input_column_names if name not in output_column_names
        ]

        dataset = dataset.map(
            self.create_conversations,
            batched=True,
            remove_columns=remove_columns,
        )

        if self._should_resize_images:
            dataset = dataset.map(self._resize_image_columns)

        split_dataset = dataset.train_test_split(
            test_size=self.split_ratio,
            seed=self.seed,
        )

        if self.is_strict_split:
            train_dataset = split_dataset["train"]
        else:
            train_dataset = dataset

        val_dataset = split_dataset["test"]

        return {
            "train": train_dataset,
            "val": val_dataset,
        }

    def create_conversations(
        self,
        examples: Dict[str, List[str]],
    ) -> Dict[str, List[Dict[str, str]]]:
        conversations = []

        for i in range(len(examples[self.data_column_name])):
            conversations.append(
                self.apply_conversation_template(
                    data=examples[self.data_column_name][i],
                )
            )

        return {
            "prompt": conversations,
        }

    def apply_conversation_template(
        self,
        data: str,
    ) -> List[Dict[str, str]]:
        conversation = [
            {
                self.role_column_name: "user",
                self.content_column_name: data,
            },
        ]
        return conversation

    def _init_resize(
        self,
        modality: str,
        max_pixels: Optional[int],
        do_resize: bool,
    ) -> None:
        self.modality = modality
        self.max_pixels = max_pixels
        self.resample_filter = getattr(
            Image,
            "LANCZOS",
            Image.Resampling.LANCZOS,
        )
        self._should_resize_images = (
            self.modality != "text"
            and self.max_pixels is not None
            and self.max_pixels > 0
            and do_resize
        )

    def _compute_target_size(
        self,
        width: int,
        height: int,
    ) -> Optional[tuple[int, int]]:
        if self.max_pixels is None:
            return None

        total_pixels = width * height
        if total_pixels <= self.max_pixels:
            return None

        scale = math.sqrt(self.max_pixels / float(total_pixels))
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        return new_width, new_height

    def _load_image_from_bytes(
        self,
        data: bytes,
    ) -> Optional[Image.Image]:
        try:
            return Image.open(io.BytesIO(data))
        except Exception:
            return None

    def _load_image(
        self,
        image: Any,
    ) -> Optional[Image.Image]:
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, (bytes, bytearray)):
            return self._load_image_from_bytes(data=bytes(image))

        if not isinstance(image, str):
            return None

        value = image.strip()
        if not value:
            return None

        if value.startswith(("http://", "https://")):
            try:
                with urllib.request.urlopen(value) as response:
                    return self._load_image_from_bytes(data=response.read())
            except Exception:
                return None

        if os.path.exists(value):
            try:
                with open(value, "rb") as f:
                    return self._load_image_from_bytes(data=f.read())
            except Exception:
                return None

        try:
            if "base64," in value:
                _, value = value.split(
                    "base64,",
                    1,
                )
            decoded = base64.b64decode(
                value,
                validate=False,
            )
            return self._load_image_from_bytes(data=decoded)
        except Exception:
            return None

    def _resize_single_image(
        self,
        image: Any,
    ) -> Any:
        if not self._should_resize_images:
            return image

        pil_image = self._load_image(image=image)
        if pil_image is None:
            return image

        target_size = self._compute_target_size(
            width=pil_image.width,
            height=pil_image.height,
        )
        if target_size is None:
            return pil_image

        try:
            return pil_image.resize(
                target_size,
                self.resample_filter,
            )
        except Exception:
            return image

    def _resize_image_columns(
        self,
        example: Dict[str, Any],
    ) -> Dict[str, Any]:
        if "images" in example and example["images"] is not None:
            if isinstance(example["images"], list):
                example["images"] = [
                    self._resize_single_image(image=img) for img in example["images"]
                ]
        if "image" in example and example["image"] is not None:
            example["image"] = self._resize_single_image(image=example["image"])
        return example


class ConversationalDataset(StructuralDataset):
    def __init__(
        self,
        data_path: str,
        split_ratio: float,
        is_strict_split: bool,
        seed: int,
        dataset_name: str,
        dataset_format: str,
        prompt_column_name: str,
        solution_column_name: str,
        reward_categories_column_name: str,
        modality: str,
        max_pixels: Optional[int],
        do_resize: bool,
    ) -> None:
        self.data_path = data_path
        self.split_ratio = split_ratio
        self.is_strict_split = is_strict_split
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.prompt_column_name = prompt_column_name
        self.solution_column_name = solution_column_name
        self.reward_categories_column_name = reward_categories_column_name
        self._init_resize(
            modality=modality,
            max_pixels=max_pixels,
            do_resize=do_resize,
        )

    def __call__(self) -> Dict[str, HFDataset]:
        file_name = f"{self.dataset_name}.{self.dataset_format}"
        full_data_path = os.path.join(
            self.data_path,
            file_name,
        )

        dataset_format = self.dataset_format
        if dataset_format == "tsv":
            dataset_format = "csv"
        if dataset_format == "jsonl":
            dataset_format = "json"

        dataset = load_dataset(
            dataset_format,
            data_files=full_data_path,
        )["train"]

        if self.prompt_column_name != "prompt":
            dataset = dataset.rename_column(
                self.prompt_column_name,
                "prompt",
            )

        output_column_names = [
            "prompt",
            "image",
            "images",
            self.solution_column_name,
            self.reward_categories_column_name,
        ]
        remove_columns = [
            column
            for column in dataset.column_names
            if column not in output_column_names
        ]

        if remove_columns:
            dataset = dataset.remove_columns(remove_columns)

        if self._should_resize_images:
            dataset = dataset.map(self._resize_image_columns)

        split_dataset = dataset.train_test_split(
            test_size=self.split_ratio,
            seed=self.seed,
        )

        if self.is_strict_split:
            train_dataset = split_dataset["train"]
        else:
            train_dataset = dataset

        val_dataset = split_dataset["test"]

        return {
            "train": train_dataset,
            "val": val_dataset,
        }
