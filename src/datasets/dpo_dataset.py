from typing import Dict, List, Tuple, Optional, Any

import importlib

datasets = importlib.import_module("datasets")
HFDataset = datasets.Dataset
load_dataset = datasets.load_dataset

import math

from PIL import Image

from ..helpers.dataset_paths import resolve_dataset_file_path
from .image_io import (
    build_image_io_settings,
    load_image,
    normalize_image_payloads,
    normalize_image_source,
)
from .image_augmentation import _build_image_augmenter


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
        chosen_column_name: str,
        rejected_column_name: str,
        role_column_name: str,
        content_column_name: str,
        assistant_column_name: str,
        modality: str,
        max_pixels: Optional[int],
        do_resize: bool,
        image_augmentation: Dict[str, Any],
        decode_image_paths: bool = False,
        dataset_subdir: Optional[str] = None,
        dataset_file_path: Optional[str] = None,
        allow_dataset_file_name_mismatch: bool = False,
        dataset_image: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data_path = data_path
        self.dataset_subdir = dataset_subdir
        self.dataset_file_path = dataset_file_path
        self.allow_dataset_file_name_mismatch = allow_dataset_file_name_mismatch
        self.split_ratio = split_ratio
        self.is_strict_split = is_strict_split
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.data_column_name = data_column_name
        self.chosen_column_name = chosen_column_name
        self.rejected_column_name = rejected_column_name
        self.role_column_name = role_column_name
        self.content_column_name = content_column_name
        self.assistant_column_name = assistant_column_name
        self.decode_image_paths = decode_image_paths
        self._init_image_io(dataset_image=dataset_image)
        self._init_resize(
            modality=modality,
            max_pixels=max_pixels,
            do_resize=do_resize,
        )
        self.image_augmenter = _build_image_augmenter(
            config=image_augmentation,
            seed=seed,
        )

    def __call__(self) -> Dict[str, HFDataset]:
        full_data_path = resolve_dataset_file_path(
            dataset_name=self.dataset_name,
            dataset_format=self.dataset_format,
            data_path=self.data_path,
            dataset_subdir=self.dataset_subdir,
            dataset_file_path=self.dataset_file_path,
            allow_dataset_file_name_mismatch=self.allow_dataset_file_name_mismatch,
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
            "chosen",
            "rejected",
        ]
        input_column_names = [
            self.data_column_name,
            self.chosen_column_name,
            self.rejected_column_name,
        ]
        remove_columns = [
            name for name in input_column_names if name not in output_column_names
        ]

        dataset = dataset.map(
            self.create_conversations,
            batched=True,
            remove_columns=remove_columns,
        )

        if self.modality != "text":
            dataset = dataset.map(self._normalize_image_columns)

        if (
            self._should_resize_images
            and self.image_augmenter is None
            and not self.decode_image_paths
        ):
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

        if self.decode_image_paths or self.image_augmenter is not None:
            train_dataset = self._decode_image_paths(
                dataset=train_dataset,
                apply_image_augmentation=self.image_augmenter is not None,
            )
            val_dataset = self._decode_image_paths(
                dataset=val_dataset,
                apply_image_augmentation=False,
            )

        return {
            "train": train_dataset,
            "val": val_dataset,
        }

    def create_conversations(
        self,
        examples: Dict[str, List[str]],
    ) -> Dict[str, List[Dict[str, str]]]:
        prompt_conversations = []
        chosen_conversations = []
        rejected_conversations = []

        for i in range(len(examples[self.data_column_name])):
            prompt_conversations.append(
                self.apply_conversation_template(
                    data=examples[self.data_column_name][i],
                    is_prompt=True,
                )
            )
            chosen_conversations.append(
                self.apply_conversation_template(
                    data=examples[self.chosen_column_name][i],
                    is_prompt=False,
                )
            )
            rejected_conversations.append(
                self.apply_conversation_template(
                    data=examples[self.rejected_column_name][i],
                    is_prompt=False,
                )
            )

        return {
            "prompt": prompt_conversations,
            "chosen": chosen_conversations,
            "rejected": rejected_conversations,
        }

    def apply_conversation_template(
        self,
        data: str,
        is_prompt: bool,
    ) -> List[Dict[str, str]]:
        if is_prompt:
            conversation = [
                {
                    self.role_column_name: "user",
                    self.content_column_name: data,
                },
            ]
        else:
            conversation = [
                {
                    self.role_column_name: self.assistant_column_name,
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

    def _init_image_io(
        self,
        dataset_image: Optional[Dict[str, Any]],
    ) -> None:
        settings = build_image_io_settings(
            dataset_image=dataset_image,
            default_image_root_dir=self.data_path,
        )
        self.image_root_dir = settings["image_root_dir"]
        self.convert_unsupported_extensions = settings["convert_unsupported_extensions"]
        self.unsupported_path_extensions = settings["unsupported_path_extensions"]
        self.converted_image_mode = settings["converted_image_mode"]

    def _compute_target_size(
        self,
        width: int,
        height: int,
    ) -> Optional[Tuple[int, int]]:
        if self.max_pixels is None:
            return None

        total_pixels = width * height
        if total_pixels <= self.max_pixels:
            return None

        scale = math.sqrt(self.max_pixels / float(total_pixels))
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        return new_width, new_height

    def _load_image(
        self,
        image: Any,
    ) -> Optional[Image.Image]:
        return load_image(
            image=image,
            image_root_dir=self.image_root_dir,
        )

    def _normalize_single_image(
        self,
        image: Any,
    ) -> Any:
        return normalize_image_source(
            image=image,
            image_root_dir=self.image_root_dir,
            convert_unsupported_extensions=self.convert_unsupported_extensions,
            unsupported_path_extensions=self.unsupported_path_extensions,
            converted_image_mode=self.converted_image_mode,
        )

    def _resize_single_image(
        self,
        image: Any,
    ) -> Any:
        if not self._should_resize_images:
            return self._normalize_single_image(image=image)

        image = self._normalize_single_image(image=image)
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

    def _process_single_image(
        self,
        image: Any,
        apply_image_augmentation: bool,
    ) -> Any:
        image = self._normalize_single_image(image=image)
        pil_image = self._load_image(image=image)
        if pil_image is None:
            if self.decode_image_paths or apply_image_augmentation:
                raise ValueError(
                    f"Failed to decode image source for DPO image processing: {repr(image)[:200]}"
                )
            return image

        if apply_image_augmentation and self.image_augmenter is not None:
            pil_image = self.image_augmenter(pil_image)

        if not self._should_resize_images:
            return pil_image

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

    def _normalize_image_columns(
        self,
        example: Dict[str, Any],
    ) -> Dict[str, Any]:
        if "images" in example and example["images"] is not None:
            if isinstance(example["images"], list):
                example["images"] = [
                    self._normalize_single_image(image=img) for img in example["images"]
                ]
        if "image" in example and example["image"] is not None:
            example["image"] = self._normalize_single_image(image=example["image"])
        for column_name in ["prompt", "chosen", "rejected"]:
            if column_name in example and example[column_name] is not None:
                example[column_name] = self._normalize_message_content(
                    content=example[column_name],
                )
        return example

    def _normalize_message_content(
        self,
        content: Any,
    ) -> Any:
        return normalize_image_payloads(
            value=content,
            image_root_dir=self.image_root_dir,
            convert_unsupported_extensions=self.convert_unsupported_extensions,
            unsupported_path_extensions=self.unsupported_path_extensions,
            converted_image_mode=self.converted_image_mode,
        )

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

    def _decode_image_value(
        self,
        image: Any,
        apply_image_augmentation: bool,
    ) -> Any:
        if isinstance(image, list):
            return [
                self._decode_image_value(
                    image=item,
                    apply_image_augmentation=apply_image_augmentation,
                )
                for item in image
            ]

        return self._process_single_image(
            image=image,
            apply_image_augmentation=apply_image_augmentation,
        )

    def _decode_image_columns(
        self,
        example: Dict[str, Any],
        apply_image_augmentation: bool,
    ) -> Dict[str, Any]:
        if "images" in example and example["images"] is not None:
            example["images"] = self._decode_image_value(
                image=example["images"],
                apply_image_augmentation=apply_image_augmentation,
            )
        if "image" in example and example["image"] is not None:
            example["image"] = self._decode_image_value(
                image=example["image"],
                apply_image_augmentation=apply_image_augmentation,
            )
        return example

    def _decode_train_image_columns(
        self,
        example: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._decode_image_columns(
            example=example,
            apply_image_augmentation=True,
        )

    def _decode_eval_image_columns(
        self,
        example: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._decode_image_columns(
            example=example,
            apply_image_augmentation=False,
        )

    def _decode_image_paths(
        self,
        dataset: HFDataset,
        apply_image_augmentation: bool,
    ) -> HFDataset:
        if apply_image_augmentation:
            return dataset.with_transform(self._decode_train_image_columns)
        return dataset.with_transform(self._decode_eval_image_columns)


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
        chosen_column_name: str,
        rejected_column_name: str,
        modality: str,
        max_pixels: Optional[int],
        do_resize: bool,
        image_augmentation: Dict[str, Any],
        decode_image_paths: bool = False,
        dataset_subdir: Optional[str] = None,
        dataset_file_path: Optional[str] = None,
        allow_dataset_file_name_mismatch: bool = False,
        dataset_image: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data_path = data_path
        self.dataset_subdir = dataset_subdir
        self.dataset_file_path = dataset_file_path
        self.allow_dataset_file_name_mismatch = allow_dataset_file_name_mismatch
        self.split_ratio = split_ratio
        self.is_strict_split = is_strict_split
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.prompt_column_name = prompt_column_name
        self.chosen_column_name = chosen_column_name
        self.rejected_column_name = rejected_column_name
        self.decode_image_paths = decode_image_paths
        self._init_image_io(dataset_image=dataset_image)
        self._init_resize(
            modality=modality,
            max_pixels=max_pixels,
            do_resize=do_resize,
        )
        self.image_augmenter = _build_image_augmenter(
            config=image_augmentation,
            seed=seed,
        )

    def __call__(self) -> Dict[str, HFDataset]:
        full_data_path = resolve_dataset_file_path(
            dataset_name=self.dataset_name,
            dataset_format=self.dataset_format,
            data_path=self.data_path,
            dataset_subdir=self.dataset_subdir,
            dataset_file_path=self.dataset_file_path,
            allow_dataset_file_name_mismatch=self.allow_dataset_file_name_mismatch,
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
        if self.chosen_column_name != "chosen":
            dataset = dataset.rename_column(
                self.chosen_column_name,
                "chosen",
            )
        if self.rejected_column_name != "rejected":
            dataset = dataset.rename_column(
                self.rejected_column_name,
                "rejected",
            )

        output_column_names = [
            "prompt",
            "chosen",
            "rejected",
            "image",
            "images",
        ]
        remove_columns = [
            column
            for column in dataset.column_names
            if column not in output_column_names
        ]

        if remove_columns:
            dataset = dataset.remove_columns(remove_columns)

        if self.modality != "text":
            dataset = dataset.map(self._normalize_image_columns)

        if (
            self._should_resize_images
            and self.image_augmenter is None
            and not self.decode_image_paths
        ):
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

        if self.decode_image_paths or self.image_augmenter is not None:
            train_dataset = self._decode_image_paths(
                dataset=train_dataset,
                apply_image_augmentation=self.image_augmenter is not None,
            )
            val_dataset = self._decode_image_paths(
                dataset=val_dataset,
                apply_image_augmentation=False,
            )

        return {
            "train": train_dataset,
            "val": val_dataset,
        }
