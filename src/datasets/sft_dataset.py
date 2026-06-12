from typing import Dict, List, Tuple, Optional, Any
import io
import math

import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoProcessor

from ..helpers.dataset_paths import resolve_dataset_file_path
from ..helpers import build_enable_thinking_kwargs
from .image_io import build_image_io_settings, load_image, normalize_image_source
from .image_augmentation import _build_image_augmenter


class StructuralDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_ratio: float,
        is_strict_split: bool,
        seed: int,
        dataset_name: str,
        dataset_format: str,
        is_sft: bool,
        is_preprocessed: bool,
        data_column_name: str,
        target_column_name: str,
        role_column_name: str,
        content_column_name: str,
        assistant_column_name: str,
        pretrained_model_name: str,
        modality: str,
        max_pixels: Optional[int],
        do_resize: bool,
        image_augmentation: Dict[str, Any],
        custom_data_encoder_path: str,
        revision: str,
        reference_data_encoder_name: str,
        left_padding: bool,
        is_enable_thinking: bool,
        max_length: int,
        sft_padding_strategy: str,
        truncation_mode: str,
        response_start_template: str,
        response_end_template: Optional[str],
        dataset_subdir: Optional[str] = None,
        dataset_file_path: Optional[str] = None,
        allow_dataset_file_name_mismatch: bool = False,
        dataset_image: Optional[Dict[str, Any]] = None,
        sft_label_mask: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data_path = data_path
        self.dataset_subdir = dataset_subdir
        self.dataset_file_path = dataset_file_path
        self.allow_dataset_file_name_mismatch = allow_dataset_file_name_mismatch
        self.split = split
        self.split_ratio = split_ratio
        self.is_strict_split = is_strict_split
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.is_sft = is_sft
        self.is_preprocessed = is_preprocessed
        self.data_column_name = data_column_name
        self.target_column_name = target_column_name
        self.role_column_name = role_column_name
        self.content_column_name = content_column_name
        self.assistant_column_name = assistant_column_name
        self.pretrained_model_name = pretrained_model_name
        self.modality = modality
        self._init_image_io(dataset_image=dataset_image)
        self._init_sft_label_mask_validation(sft_label_mask=sft_label_mask)
        self._init_resize(
            modality=modality,
            max_pixels=max_pixels,
            do_resize=do_resize,
        )
        self.image_augmenter = None
        if split == "train":
            self.image_augmenter = _build_image_augmenter(
                config=image_augmentation,
                seed=seed,
            )

        if is_preprocessed:
            data_encoder_path = custom_data_encoder_path
        else:
            data_encoder_path = self.pretrained_model_name

        if self.modality == "text":
            self.data_encoder = AutoTokenizer.from_pretrained(
                data_encoder_path,
                use_fast=True,
                revision=revision,
            )

            if self.data_encoder.chat_template is None:
                reference_data_encoder = AutoTokenizer.from_pretrained(
                    reference_data_encoder_name
                )
                self.data_encoder.chat_template = reference_data_encoder.chat_template

            if self.data_encoder.pad_token_id is None:
                self.data_encoder.pad_token_id = self.data_encoder.eos_token_id

            if left_padding:
                self.data_encoder.padding_side = "left"
            else:
                self.data_encoder.padding_side = "right"
            self._init_sft_padding(
                sft_padding_strategy=sft_padding_strategy,
                truncation_mode=truncation_mode,
            )
        else:
            self.data_encoder = AutoProcessor.from_pretrained(
                data_encoder_path,
                revision=revision,
            )

            if self.data_encoder.tokenizer.chat_template is None:
                reference_data_encoder = AutoTokenizer.from_pretrained(
                    reference_data_encoder_name
                )
                self.data_encoder.tokenizer.chat_template = (
                    reference_data_encoder.chat_template
                )

            if self.data_encoder.tokenizer.pad_token_id is None:
                self.data_encoder.tokenizer.pad_token_id = (
                    self.data_encoder.tokenizer.eos_token_id
                )

            if left_padding:
                self.data_encoder.tokenizer.padding_side = "left"
            else:
                self.data_encoder.tokenizer.padding_side = "right"
            self._init_sft_padding(
                sft_padding_strategy=sft_padding_strategy,
                truncation_mode=truncation_mode,
            )

        self.is_enable_thinking = is_enable_thinking

        dataset = self.get_dataset()
        self.datas = dataset["datas"]
        self.labels = dataset["labels"]
        self.max_length = max_length

        self.response_start_tokens = self.data_encoder(
            text=response_start_template,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0)
        self.eos_token = (
            self.data_encoder.eos_token
            if modality == "text"
            else self.data_encoder.tokenizer.eos_token
        )
        response_end_text = (
            self.eos_token if response_end_template is None else response_end_template
        )
        self.response_end_tokens = self.data_encoder(
            text=response_end_text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0)
        self.ignore_index = -100

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        prompt = self.apply_chat_template(
            data=self.datas[idx],
            label=self.labels[idx],
        )

        image = None
        if self.modality != "text":
            image = []
            for part in self.datas[idx]:
                if isinstance(part, dict) and part.get("type") == "image":
                    image.append(part.get("image"))

            if not image:
                image = None
            elif self.image_augmenter is not None or self._should_resize_images:
                image = [self._process_single_image(image=img) for img in image]
            else:
                image = [self._normalize_single_image(image=img) for img in image]

        encoded = self.encode_data(
            data=prompt,
            image=image,
        )
        if self.is_sft:
            encoded = self.add_sft_label(encoded=encoded)
        if "token_type_ids" in encoded.keys():
            del encoded["token_type_ids"]
        if not "labels" in encoded.keys():
            encoded["labels"] = encoded["input_ids"]

        return encoded

    def get_dataset(self) -> Dict[str, List[Any]]:
        if self.split in ["train", "val"]:
            full_data_path = resolve_dataset_file_path(
                dataset_name=self.dataset_name,
                dataset_format=self.dataset_format,
                data_path=self.data_path,
                dataset_subdir=self.dataset_subdir,
                dataset_file_path=self.dataset_file_path,
                allow_dataset_file_name_mismatch=self.allow_dataset_file_name_mismatch,
            )
        else:
            raise ValueError(f"Inavalid split: {self.split}")

        if self.dataset_format == "parquet":
            data = pd.read_parquet(full_data_path)
        elif self.dataset_format in ["json", "jsonl"]:
            data = pd.read_json(
                full_data_path,
                lines=True if self.dataset_format == "jsonl" else False,
            )
        elif self.dataset_format in ["csv", "tsv"]:
            data = pd.read_csv(
                full_data_path,
                sep="\t" if self.dataset_format == "tsv" else None,
            )
        else:
            raise ValueError(f"Unsupported dataset format: {self.dataset_format}")

        data = data.fillna("_")

        train_data, val_data = train_test_split(
            data,
            test_size=self.split_ratio,
            random_state=self.seed,
            shuffle=True,
        )
        if self.split == "train" and self.is_strict_split:
            data = train_data
        if self.split == "val":
            data = val_data

        datas = data[self.data_column_name].tolist()
        labels = data[self.target_column_name].apply(lambda x: x.strip()).tolist()
        return {
            "datas": datas,
            "labels": labels,
        }

    def apply_chat_template(
        self,
        data: str,
        label: str,
    ) -> str:
        if self.modality == "text":
            conversation = [
                {
                    self.role_column_name: "user",
                    self.content_column_name: data,
                },
                {
                    self.role_column_name: self.assistant_column_name,
                    self.content_column_name: label,
                },
            ]
        else:
            conversation = [
                {
                    self.role_column_name: "user",
                    self.content_column_name: data,
                },
                {
                    self.role_column_name: self.assistant_column_name,
                    self.content_column_name: [
                        {
                            "type": "text",
                            "text": label,
                        },
                    ],
                },
            ]

        chat_template_kwargs = build_enable_thinking_kwargs(
            data_encoder=self.data_encoder,
            is_enable_thinking=self.is_enable_thinking,
        )
        prompt = self.data_encoder.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=False,
            **chat_template_kwargs,
        )
        return prompt

    def encode_data(
        self,
        data: str,
        image: Optional[List[Any]],
    ) -> Dict[str, torch.Tensor]:
        kwargs = {
            "text": data,
            "padding": self._get_encode_padding(),
            "max_length": self.max_length,
            "truncation": True,
            "return_tensors": "pt",
            "add_special_tokens": True,
        }
        if self.modality != "text":
            kwargs["images"] = image
        encoded = self.data_encoder(**kwargs)

        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        if self.sft_padding_strategy == "dynamic" and "attention_mask" not in encoded:
            raise ValueError(
                "SFT dynamic padding requires attention_mask in encoded data."
            )
        return encoded

    def add_sft_label(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        input_ids = encoded["input_ids"]
        labels = torch.full_like(
            input_ids,
            self.ignore_index,
        )

        start_indices = self.find_pattern_indices(
            input_ids=input_ids,
            pattern=self.response_start_tokens,
        )
        end_indices = self.find_pattern_indices(
            input_ids=input_ids,
            pattern=self.response_end_tokens,
        )

        end_idx_pos = 0
        for start_idx in start_indices:
            content_start = start_idx + len(self.response_start_tokens)

            while end_idx_pos < len(end_indices):
                if end_indices[end_idx_pos] > start_idx:
                    content_end = end_indices[end_idx_pos]
                    labels[content_start:content_end] = input_ids[
                        content_start:content_end
                    ]
                    end_idx_pos += 1
                    break
                end_idx_pos += 1

            else:
                labels[content_start:] = input_ids[content_start:]

        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            labels = labels.masked_fill(
                attention_mask == 0,
                self.ignore_index,
            )

        encoded["labels"] = labels
        self._validate_sft_label_mask(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            start_indices=start_indices,
            end_indices=end_indices,
        )
        return encoded

    def validate_sft_label_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        if "input_ids" not in batch or "labels" not in batch:
            raise ValueError("batch must contain input_ids and labels.")

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        reports = []
        for sample_idx in range(input_ids.size(0)):
            sample_attention_mask = None
            if attention_mask is not None:
                sample_attention_mask = attention_mask[sample_idx]
            reports.append(
                self._build_sft_label_mask_report(
                    input_ids=input_ids[sample_idx],
                    labels=labels[sample_idx],
                    attention_mask=sample_attention_mask,
                    start_indices=self.find_pattern_indices(
                        input_ids=input_ids[sample_idx],
                        pattern=self.response_start_tokens,
                    ),
                    end_indices=self.find_pattern_indices(
                        input_ids=input_ids[sample_idx],
                        pattern=self.response_end_tokens,
                    ),
                )
            )

        errors = [error for report in reports for error in report["errors"]]
        if errors and self.sft_label_mask_validation_mode == "strict":
            raise ValueError(
                "SFT batch label mask validation failed: " + "; ".join(errors)
            )
        return {
            "num_samples": len(reports),
            "reports": reports,
            "errors": errors,
        }

    def _validate_sft_label_mask(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        start_indices: List[int],
        end_indices: List[int],
    ) -> None:
        if not self.sft_label_mask_validation_enabled:
            return

        report = self._build_sft_label_mask_report(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            start_indices=start_indices,
            end_indices=end_indices,
        )
        if report["errors"] and self.sft_label_mask_validation_mode == "strict":
            raise ValueError(
                "SFT label mask validation failed: " + "; ".join(report["errors"])
            )

    def _build_sft_label_mask_report(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        start_indices: List[int],
        end_indices: List[int],
    ) -> Dict[str, Any]:
        expected_mask = self._build_sft_expected_label_mask(
            input_ids=input_ids,
            start_indices=start_indices,
            end_indices=end_indices,
        )
        if attention_mask is not None:
            expected_mask = expected_mask & (attention_mask != 0)

        active_label_mask = labels != self.ignore_index
        assistant_label_tokens = int(active_label_mask.sum().item())
        prompt_label_leaks = int((active_label_mask & ~expected_mask).sum().item())
        target_label_mismatches = int(
            ((labels != input_ids) & expected_mask).sum().item()
        )
        padding_label_leaks = 0
        if attention_mask is not None:
            padding_label_leaks = int(
                (active_label_mask & (attention_mask == 0)).sum().item()
            )

        errors = []
        if len(start_indices) == 0:
            errors.append("response_start_template was not found.")
        if assistant_label_tokens < self.sft_label_mask_min_assistant_label_tokens:
            errors.append(
                "assistant label token count is below "
                f"{self.sft_label_mask_min_assistant_label_tokens}."
            )
        if prompt_label_leaks > 0:
            errors.append(f"prompt label leak count={prompt_label_leaks}.")
        if padding_label_leaks > 0:
            errors.append(f"padding label leak count={padding_label_leaks}.")
        if target_label_mismatches > 0:
            errors.append(f"target label mismatch count={target_label_mismatches}.")

        report = {
            "assistant_label_tokens": assistant_label_tokens,
            "response_start_count": len(start_indices),
            "response_end_count": len(end_indices),
            "prompt_label_leaks": prompt_label_leaks,
            "padding_label_leaks": padding_label_leaks,
            "target_label_mismatches": target_label_mismatches,
            "truncated_assistant": self._is_assistant_truncated(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_indices=start_indices,
                end_indices=end_indices,
            ),
            "errors": errors,
        }
        return report

    def _build_sft_expected_label_mask(
        self,
        input_ids: torch.Tensor,
        start_indices: List[int],
        end_indices: List[int],
    ) -> torch.Tensor:
        expected_mask = torch.zeros_like(
            input_ids,
            dtype=torch.bool,
        )

        end_idx_pos = 0
        for start_idx in start_indices:
            content_start = start_idx + len(self.response_start_tokens)
            while end_idx_pos < len(end_indices):
                if end_indices[end_idx_pos] > start_idx:
                    content_end = end_indices[end_idx_pos]
                    expected_mask[content_start:content_end] = True
                    end_idx_pos += 1
                    break
                end_idx_pos += 1
            else:
                expected_mask[content_start:] = True

        return expected_mask

    def _is_assistant_truncated(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        start_indices: List[int],
        end_indices: List[int],
    ) -> bool:
        if not self.sft_label_mask_report_truncated_assistant:
            return False
        if len(start_indices) == 0:
            return False
        if any(end_idx > start_indices[-1] for end_idx in end_indices):
            return False
        if attention_mask is None:
            return input_ids.size(0) >= self.max_length
        return int(attention_mask.sum().item()) >= self.max_length

    def find_pattern_indices(
        self,
        input_ids: torch.Tensor,
        pattern: torch.Tensor,
    ) -> List[int]:
        pattern_length = pattern.size(0)

        if pattern_length > input_ids.size(0):
            return []

        indices = []
        for i in range(input_ids.size(0) - pattern_length + 1):
            if torch.equal(input_ids[i : i + pattern_length], pattern):
                indices.append(i)
        return indices

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

    def _init_sft_label_mask_validation(
        self,
        sft_label_mask: Optional[Dict[str, Any]],
    ) -> None:
        config = sft_label_mask or {}
        self.sft_label_mask_validation_enabled = bool(
            self._get_sft_label_mask_config_value(
                config=config,
                key="validation_enabled",
                default=True,
            )
        )
        self.sft_label_mask_validation_mode = str(
            self._get_sft_label_mask_config_value(
                config=config,
                key="validation_mode",
                default="strict",
            )
        )
        if self.sft_label_mask_validation_mode not in ["strict", "report"]:
            raise ValueError("sft_label_mask.validation_mode must be strict or report.")
        self.sft_label_mask_min_assistant_label_tokens = int(
            self._get_sft_label_mask_config_value(
                config=config,
                key="min_assistant_label_tokens",
                default=1,
            )
        )
        self.sft_label_mask_report_truncated_assistant = bool(
            self._get_sft_label_mask_config_value(
                config=config,
                key="report_truncated_assistant",
                default=True,
            )
        )

    @staticmethod
    def _get_sft_label_mask_config_value(
        config: Dict[str, Any],
        key: str,
        default: Any,
    ) -> Any:
        if key in config:
            return config[key]
        return default

    def _init_sft_padding(
        self,
        sft_padding_strategy: str,
        truncation_mode: str,
    ) -> None:
        if sft_padding_strategy not in ["max_length", "dynamic"]:
            raise ValueError("sft_padding_strategy must be max_length or dynamic.")
        if truncation_mode not in ["keep_start", "keep_end"]:
            raise ValueError("truncation_mode must be keep_start or keep_end.")

        self.sft_padding_strategy = sft_padding_strategy
        self.truncation_mode = truncation_mode
        truncation_side = "right" if truncation_mode == "keep_start" else "left"
        if self.modality == "text":
            self.data_encoder.truncation_side = truncation_side
        else:
            self.data_encoder.tokenizer.truncation_side = truncation_side

    def _get_encode_padding(
        self,
    ) -> Any:
        if self.sft_padding_strategy == "max_length":
            return "max_length"
        return False

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

    def _process_single_image(
        self,
        image: Any,
    ) -> Any:
        image = self._normalize_single_image(image=image)
        if self.image_augmenter is None and not self._should_resize_images:
            return image

        pil_image = self._load_image(image=image)
        if pil_image is None:
            return image

        if self.image_augmenter is not None:
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


class ConversationalDataset(StructuralDataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_ratio: float,
        is_strict_split: bool,
        seed: int,
        dataset_name: str,
        dataset_format: str,
        is_sft: bool,
        is_preprocessed: bool,
        conversation_column_name: str,
        role_column_name: str,
        content_column_name: str,
        pretrained_model_name: str,
        modality: str,
        max_pixels: Optional[int],
        do_resize: bool,
        image_augmentation: Dict[str, Any],
        custom_data_encoder_path: str,
        revision: str,
        reference_data_encoder_name: str,
        left_padding: bool,
        is_enable_thinking: bool,
        max_length: int,
        sft_padding_strategy: str,
        truncation_mode: str,
        response_start_template: str,
        response_end_template: Optional[str],
        dataset_subdir: Optional[str] = None,
        dataset_file_path: Optional[str] = None,
        allow_dataset_file_name_mismatch: bool = False,
        dataset_image: Optional[Dict[str, Any]] = None,
        sft_label_mask: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data_path = data_path
        self.dataset_subdir = dataset_subdir
        self.dataset_file_path = dataset_file_path
        self.allow_dataset_file_name_mismatch = allow_dataset_file_name_mismatch
        self.split = split
        self.split_ratio = split_ratio
        self.is_strict_split = is_strict_split
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.is_sft = is_sft
        self.is_preprocessed = is_preprocessed
        self.conversation_column_name = conversation_column_name
        self.role_column_name = role_column_name
        self.content_column_name = content_column_name
        self.pretrained_model_name = pretrained_model_name
        self.modality = modality
        self._init_image_io(dataset_image=dataset_image)
        self._init_sft_label_mask_validation(sft_label_mask=sft_label_mask)
        self._init_resize(
            modality=modality,
            max_pixels=max_pixels,
            do_resize=do_resize,
        )
        self.image_augmenter = None
        if split == "train":
            self.image_augmenter = _build_image_augmenter(
                config=image_augmentation,
                seed=seed,
            )

        if is_preprocessed:
            data_encoder_path = custom_data_encoder_path
        else:
            data_encoder_path = self.pretrained_model_name

        if self.modality == "text":
            self.data_encoder = AutoTokenizer.from_pretrained(
                data_encoder_path,
                use_fast=True,
                revision=revision,
            )

            if self.data_encoder.chat_template is None:
                reference_data_encoder = AutoTokenizer.from_pretrained(
                    reference_data_encoder_name
                )
                self.data_encoder.chat_template = reference_data_encoder.chat_template

            if self.data_encoder.pad_token_id is None:
                self.data_encoder.pad_token_id = self.data_encoder.eos_token_id

            if left_padding:
                self.data_encoder.padding_side = "left"
            else:
                self.data_encoder.padding_side = "right"
            self._init_sft_padding(
                sft_padding_strategy=sft_padding_strategy,
                truncation_mode=truncation_mode,
            )
        else:
            self.data_encoder = AutoProcessor.from_pretrained(
                data_encoder_path,
                revision=revision,
            )

            if self.data_encoder.tokenizer.chat_template is None:
                reference_data_encoder = AutoTokenizer.from_pretrained(
                    reference_data_encoder_name
                )
                self.data_encoder.tokenizer.chat_template = (
                    reference_data_encoder.chat_template
                )

            if self.data_encoder.tokenizer.pad_token_id is None:
                self.data_encoder.tokenizer.pad_token_id = (
                    self.data_encoder.tokenizer.eos_token_id
                )

            if left_padding:
                self.data_encoder.tokenizer.padding_side = "left"
            else:
                self.data_encoder.tokenizer.padding_side = "right"
            self._init_sft_padding(
                sft_padding_strategy=sft_padding_strategy,
                truncation_mode=truncation_mode,
            )

        self.is_enable_thinking = is_enable_thinking

        dataset = self.get_dataset()
        self.conversations = dataset["conversations"]
        self.max_length = max_length

        self.response_start_tokens = self.data_encoder(
            text=response_start_template,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0)
        self.eos_token = (
            self.data_encoder.eos_token
            if modality == "text"
            else self.data_encoder.tokenizer.eos_token
        )
        response_end_text = (
            self.eos_token if response_end_template is None else response_end_template
        )
        self.response_end_tokens = self.data_encoder(
            text=response_end_text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0)
        self.ignore_index = -100

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        prompt = self.apply_chat_template(
            conversation=self.conversations[idx],
        )

        image = None
        if self.modality != "text":
            image = []
            for turn in self.conversations[idx]:
                content = turn[self.content_column_name]
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "image":
                            image.append(part.get("image"))

            if not image:
                image = None
            elif self.image_augmenter is not None or self._should_resize_images:
                image = [self._process_single_image(image=img) for img in image]
            else:
                image = [self._normalize_single_image(image=img) for img in image]

        encoded = self.encode_data(
            data=prompt,
            image=image,
        )
        if self.is_sft:
            encoded = self.add_sft_label(encoded=encoded)
        if "token_type_ids" in encoded.keys():
            del encoded["token_type_ids"]
        if not "labels" in encoded.keys():
            encoded["labels"] = encoded["input_ids"]
        return encoded

    def get_dataset(self) -> Dict[str, List[Any]]:
        if self.split in ["train", "val"]:
            full_data_path = resolve_dataset_file_path(
                dataset_name=self.dataset_name,
                dataset_format=self.dataset_format,
                data_path=self.data_path,
                dataset_subdir=self.dataset_subdir,
                dataset_file_path=self.dataset_file_path,
                allow_dataset_file_name_mismatch=self.allow_dataset_file_name_mismatch,
            )
        else:
            raise ValueError(f"Inavalid split: {self.split}")

        if self.dataset_format == "parquet":
            data = pd.read_parquet(full_data_path)
        elif self.dataset_format in ["json", "jsonl"]:
            data = pd.read_json(
                full_data_path,
                lines=True if self.dataset_format == "jsonl" else False,
            )
        elif self.dataset_format in ["csv", "tsv"]:
            data = pd.read_csv(
                full_data_path,
                sep="\t" if self.dataset_format == "tsv" else None,
            )
        else:
            raise ValueError(f"Unsupported dataset format: {self.dataset_format}")

        data = data.fillna("_")

        train_data, val_data = train_test_split(
            data,
            test_size=self.split_ratio,
            random_state=self.seed,
            shuffle=True,
        )
        if self.split == "train" and self.is_strict_split:
            data = train_data
        if self.split == "val":
            data = val_data

        conversations = data[self.conversation_column_name].tolist()
        return {
            "conversations": conversations,
        }

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
    ) -> str:
        preprocessed_conversation = []
        for turn in conversation:
            preprocessed_turn = {
                self.role_column_name: turn[self.role_column_name],
                self.content_column_name: turn[self.content_column_name],
            }
            preprocessed_conversation.append(preprocessed_turn)

        chat_template_kwargs = build_enable_thinking_kwargs(
            data_encoder=self.data_encoder,
            is_enable_thinking=self.is_enable_thinking,
        )
        prompt = self.data_encoder.apply_chat_template(
            conversation=preprocessed_conversation,
            tokenize=False,
            add_generation_prompt=False,
            **chat_template_kwargs,
        )
        return prompt
