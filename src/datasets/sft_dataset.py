from typing import Dict, List, Tuple, Union, Optional, Any

import importlib

datasets = importlib.import_module("datasets")
HFDataset = datasets.Dataset

import io
import math
import re

from omegaconf import DictConfig, ListConfig

import numpy as np

from sklearn.model_selection import train_test_split
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoProcessor

from ..helpers.dataset_paths import (
    resolve_dataset_file_specs,
    resolve_optional_dataset_file_specs,
)
from ..helpers import build_enable_thinking_kwargs
from .dataset_loading import (
    load_pandas_dataset_specs,
    load_weighted_pandas_dataset_specs,
)
from .image_io import build_image_io_settings, load_image, normalize_image_source
from .image_augmentation import build_image_augmenter


class SFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_ratio: float,
        is_strict_split: bool,
        seed: int,
        dataset_name: str,
        dataset_format: str,
        data_type: str,
        is_sft: bool,
        is_preprocessed: bool,
        data_column_name: str,
        conversation_column_name: str,
        target_column_name: str,
        role_column_name: str,
        content_column_name: str,
        assistant_column_name: str,
        pretrained_model_name: str,
        modality: str,
        max_pixels: Optional[int],
        do_resize: bool,
        image_augmentation: Union[Dict[str, Any], DictConfig],
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
        dataset_subdir: Optional[str],
        dataset_file_path: Optional[str],
        dataset_file_paths: Optional[Union[List[str], ListConfig]],
        dataset_files: Optional[Union[List[Dict[str, Any]], ListConfig]],
        allow_dataset_file_name_mismatch: bool,
        val_dataset_file_path: Optional[str],
        val_dataset_file_paths: Optional[Union[List[str], ListConfig]],
        val_dataset_files: Optional[Union[List[Dict[str, Any]], ListConfig]],
        allow_val_dataset_file_name_mismatch: bool,
        dataset_resampling: DictConfig,
        dataset_image: Optional[Union[Dict[str, Any], DictConfig]],
        sft_label_mask_backend: str,
        sft_label_mask_validation_enabled: bool,
        sft_label_mask_validation_mode: str,
        sft_label_mask_min_assistant_label_tokens: int,
        sft_label_mask_report_truncated_assistant: bool,
    ) -> None:
        if data_type not in {"structural", "conversational"}:
            raise ValueError("data_type must be structural or conversational.")
        self.data_type = data_type
        self.data_path = data_path
        self.dataset_subdir = dataset_subdir
        self.dataset_file_path = dataset_file_path
        self.dataset_file_paths = dataset_file_paths
        self.dataset_files = dataset_files
        self.allow_dataset_file_name_mismatch = allow_dataset_file_name_mismatch
        self.val_dataset_file_path = val_dataset_file_path
        self.val_dataset_file_paths = val_dataset_file_paths
        self.val_dataset_files = val_dataset_files
        self.allow_val_dataset_file_name_mismatch = allow_val_dataset_file_name_mismatch
        self.dataset_resampling = dataset_resampling
        self.split = split
        self.split_ratio = split_ratio
        self.is_strict_split = is_strict_split
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.is_sft = is_sft
        self.is_preprocessed = is_preprocessed
        self.data_column_name = data_column_name
        self.conversation_column_name = conversation_column_name
        self.target_column_name = target_column_name
        self.role_column_name = role_column_name
        self.content_column_name = content_column_name
        self.assistant_column_name = assistant_column_name
        self.pretrained_model_name = pretrained_model_name
        self.modality = modality
        self._init_image_io(dataset_image=dataset_image)
        self._init_sft_label_mask_validation(
            backend=sft_label_mask_backend,
            validation_enabled=sft_label_mask_validation_enabled,
            validation_mode=sft_label_mask_validation_mode,
            min_assistant_label_tokens=sft_label_mask_min_assistant_label_tokens,
            report_truncated_assistant=sft_label_mask_report_truncated_assistant,
        )
        self._init_resize(
            modality=modality,
            max_pixels=max_pixels,
            do_resize=do_resize,
        )
        self.image_augmenter = None
        if split == "train":
            self.image_augmenter = build_image_augmenter(
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
        self.column_names = ["input_ids", "attention_mask", "labels"]

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
        return len(self.datas)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        conversation = self._build_conversation(
            data=self.datas[idx],
            label=self.labels[idx] if self.labels is not None else None,
        )
        if self.sft_label_mask_backend == "native":
            return self._encode_native_conversation(conversation=conversation)
        prompt = self.apply_chat_template(conversation=conversation)
        image = None
        if self.modality != "text":
            image = []
            for turn in conversation:
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

    def to_hf_dataset(self) -> HFDataset:
        return HFDataset.from_dict({"_index": range(len(self))}).with_transform(
            _SFTDatasetTransform(dataset=self)
        )

    def get_dataset(self) -> Dict[str, Optional[List[Any]]]:
        val_data_specs = resolve_optional_dataset_file_specs(
            dataset_name=self.dataset_name,
            dataset_format=self.dataset_format,
            data_path=self.data_path,
            dataset_file_path=self.val_dataset_file_path,
            dataset_file_paths=self.val_dataset_file_paths,
            dataset_files=self.val_dataset_files,
            allow_dataset_file_name_mismatch=self.allow_val_dataset_file_name_mismatch,
            path_label="val_dataset",
            allow_weight=False,
        )

        if self.split == "val" and val_data_specs is not None:
            data = load_pandas_dataset_specs(
                dataset_file_specs=val_data_specs,
            )
        elif self.split in ["train", "val"]:
            train_data_specs = resolve_dataset_file_specs(
                dataset_name=self.dataset_name,
                dataset_format=self.dataset_format,
                data_path=self.data_path,
                dataset_subdir=self.dataset_subdir,
                dataset_file_path=self.dataset_file_path,
                dataset_file_paths=self.dataset_file_paths,
                dataset_files=self.dataset_files,
                allow_dataset_file_name_mismatch=self.allow_dataset_file_name_mismatch,
                path_label="dataset",
                allow_weight=True,
            )
            if self.dataset_resampling.enabled:
                data = load_weighted_pandas_dataset_specs(
                    dataset_file_specs=train_data_specs,
                    dataset_resampling=self.dataset_resampling,
                    seed=self.seed,
                )
            else:
                data = load_pandas_dataset_specs(
                    dataset_file_specs=train_data_specs,
                )
        else:
            raise ValueError(f"Inavalid split: {self.split}")

        data = data.fillna("_")

        if val_data_specs is None:
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

        if self.data_type == "conversational":
            datas = [
                _normalize_sft_nested_value(value=value)
                for value in data[self.conversation_column_name].tolist()
            ]
            labels = None
        else:
            datas = [
                _normalize_sft_nested_value(value=value)
                for value in data[self.data_column_name].tolist()
            ]
            labels = data[self.target_column_name].apply(lambda x: x.strip()).tolist()
        return {
            "datas": datas,
            "labels": labels,
        }

    def apply_chat_template(
        self,
        conversation: List[Dict[str, Any]],
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

        start_indices = self._find_pattern_indices(
            input_ids=input_ids,
            pattern=self.response_start_tokens,
        )
        end_indices = self._find_pattern_indices(
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
            assistant_mask=None,
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
        assistant_masks = None
        if self.sft_label_mask_backend == "native":
            assistant_masks = batch.get("assistant_masks")
            if (
                assistant_masks is None
                or assistant_masks.shape != input_ids.shape
                or not torch.all((assistant_masks == 0) | (assistant_masks == 1))
            ):
                raise ValueError(
                    "Native batch validation requires aligned assistant_masks."
                )
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
                    start_indices=self._find_pattern_indices(
                        input_ids=input_ids[sample_idx],
                        pattern=self.response_start_tokens,
                    ),
                    end_indices=self._find_pattern_indices(
                        input_ids=input_ids[sample_idx],
                        pattern=self.response_end_tokens,
                    ),
                    assistant_mask=(
                        assistant_masks[sample_idx]
                        if assistant_masks is not None
                        else None
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

    def _build_conversation(
        self,
        data: Any,
        label: Optional[str],
    ) -> List[Dict[str, Any]]:
        if self.data_type == "conversational":
            return data
        content = label
        if self.modality != "text":
            content = [
                {
                    "type": "text",
                    "text": label,
                },
            ]
        return [
            {
                self.role_column_name: "user",
                self.content_column_name: data,
            },
            {
                self.role_column_name: self.assistant_column_name,
                self.content_column_name: content,
            },
        ]

    def _encode_native_conversation(
        self,
        conversation: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        tokenizer = (
            self.data_encoder
            if self.modality == "text"
            else self.data_encoder.tokenizer
        )
        if not tokenizer.is_fast:
            raise ValueError("Native assistant masking requires a fast tokenizer.")
        template = (
            tokenizer.get_chat_template()
            if self.modality == "text"
            else self.data_encoder.chat_template
        )
        if isinstance(template, dict):
            template = template.get("default")
        if not isinstance(template, str) or not re.search(
            r"\{%[-+]?\s*generation\s*[-+]?%\}", template
        ):
            raise ValueError(
                "Native assistant masking requires a template with generation tags."
            )

        messages = []
        for turn in conversation:
            content = turn[self.content_column_name]
            if isinstance(content, list):
                content = [dict(part) for part in content]
                for part in content:
                    if part.get("type") == "image":
                        part["image"] = self._process_single_image(image=part["image"])
            messages.append(
                {
                    self.role_column_name: turn[self.role_column_name],
                    self.content_column_name: content,
                },
            )
        encode_kwargs = {
            "padding": self._get_encode_padding(),
            "max_length": self.max_length,
            "truncation": True,
        }
        template_kwargs = build_enable_thinking_kwargs(
            data_encoder=self.data_encoder,
            is_enable_thinking=self.is_enable_thinking,
        )
        if self.modality == "text":
            template_kwargs.update(encode_kwargs)
        else:
            template_kwargs["processor_kwargs"] = encode_kwargs
        encoded = self.data_encoder.apply_chat_template(
            conversation=messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
            return_tensors="pt",
            **template_kwargs,
        )
        assistant_mask = encoded.pop(
            "assistant_masks",
            None,
        )
        if assistant_mask is None:
            raise ValueError("Native assistant masking did not return assistant_masks.")
        encoded = {key: value.squeeze(0) for key, value in encoded.items()}
        assistant_mask = torch.as_tensor(assistant_mask).squeeze(0)
        input_ids = encoded["input_ids"]
        if assistant_mask.shape != input_ids.shape or not torch.all(
            (assistant_mask == 0) | (assistant_mask == 1)
        ):
            raise ValueError(
                "Native assistant_masks must be binary and aligned with input_ids."
            )
        attention_mask = encoded["attention_mask"]
        label_mask = (
            assistant_mask.bool()
            if self.is_sft
            else torch.ones_like(
                input_ids,
                dtype=torch.bool,
            )
        )
        encoded["labels"] = input_ids.masked_fill(
            ~(label_mask & attention_mask.bool()),
            self.ignore_index,
        )
        if self.is_sft:
            self._validate_sft_label_mask(
                input_ids=input_ids,
                labels=encoded["labels"],
                attention_mask=attention_mask,
                start_indices=[],
                end_indices=[],
                assistant_mask=assistant_mask,
            )
        encoded.pop(
            "token_type_ids",
            None,
        )
        return encoded

    def _validate_sft_label_mask(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        start_indices: List[int],
        end_indices: List[int],
        assistant_mask: Optional[torch.Tensor],
    ) -> None:
        if not self.sft_label_mask_validation_enabled:
            return

        report = self._build_sft_label_mask_report(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            start_indices=start_indices,
            end_indices=end_indices,
            assistant_mask=assistant_mask,
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
        assistant_mask: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        expected_mask = (
            assistant_mask.bool()
            if assistant_mask is not None
            else self._build_sft_expected_label_mask(
                input_ids=input_ids,
                start_indices=start_indices,
                end_indices=end_indices,
            )
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
        if assistant_mask is None and len(start_indices) == 0:
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
                assistant_mask=assistant_mask,
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
        assistant_mask: Optional[torch.Tensor],
    ) -> bool:
        if not self.sft_label_mask_report_truncated_assistant:
            return False
        if assistant_mask is not None:
            active_indices = (
                torch.nonzero(attention_mask).flatten()
                if attention_mask is not None
                else torch.arange(input_ids.size(0))
            )
            if active_indices.numel() < self.max_length:
                return False
            last_index = int(active_indices[-1].item())
            tokenizer = (
                self.data_encoder
                if self.modality == "text"
                else self.data_encoder.tokenizer
            )
            return bool(assistant_mask[last_index].item()) and (
                int(input_ids[last_index].item()) != tokenizer.eos_token_id
            )
        if len(start_indices) == 0:
            return False
        if any(end_idx > start_indices[-1] for end_idx in end_indices):
            return False
        if attention_mask is None:
            return input_ids.size(0) >= self.max_length
        return int(attention_mask.sum().item()) >= self.max_length

    def _find_pattern_indices(
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
        dataset_image: Optional[Union[Dict[str, Any], DictConfig]],
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
        backend: str,
        validation_enabled: bool,
        validation_mode: str,
        min_assistant_label_tokens: int,
        report_truncated_assistant: bool,
    ) -> None:
        self.sft_label_mask_backend = backend
        if self.sft_label_mask_backend not in {"repo", "native"}:
            raise ValueError("sft_label_mask.backend must be repo or native.")
        self.sft_label_mask_validation_enabled = validation_enabled
        self.sft_label_mask_validation_mode = validation_mode
        if self.sft_label_mask_validation_mode not in ["strict", "report"]:
            raise ValueError("sft_label_mask.validation_mode must be strict or report.")
        self.sft_label_mask_min_assistant_label_tokens = int(min_assistant_label_tokens)
        self.sft_label_mask_report_truncated_assistant = report_truncated_assistant

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
        new_width = max(
            1,
            int(width * scale),
        )
        new_height = max(
            1,
            int(height * scale),
        )
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
            converted_image_mode=self.converted_image_mode,
        )

    def _normalize_single_image(
        self,
        image: Any,
    ) -> Any:
        normalized_image = normalize_image_source(
            image=image,
            image_root_dir=self.image_root_dir,
            convert_unsupported_extensions=self.convert_unsupported_extensions,
            unsupported_path_extensions=self.unsupported_path_extensions,
            converted_image_mode=self.converted_image_mode,
        )
        if isinstance(normalized_image, (bytes, bytearray, dict)):
            loaded_image = self._load_image(image=normalized_image)
            if loaded_image is not None:
                return loaded_image
        return normalized_image

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


def _normalize_sft_nested_value(
    value: Any,
) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        return [_normalize_sft_nested_value(value=item) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalize_sft_nested_value(value=item)
            for key, item in value.items()
            if item is not None
        }
    return value


class _SFTDatasetTransform:
    def __init__(
        self,
        dataset: Dataset,
    ) -> None:
        self.dataset = dataset

    def __call__(
        self,
        batch: Dict[str, List[int]],
    ) -> Dict[str, List[Any]]:
        examples = [self.dataset[index] for index in batch["_index"]]
        return {key: [example[key] for example in examples] for key in examples[0]}
