from typing import Dict, List, Optional, Callable, Any

import copy
import importlib
import math

datasets = importlib.import_module("datasets")
HFDataset = datasets.Dataset

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoProcessor

from ..helpers.dataset_paths import (
    resolve_dataset_file_specs,
    resolve_optional_dataset_file_specs,
)
from ..helpers import build_enable_thinking_kwargs
from .dataset_loading import load_hf_dataset_specs, load_weighted_hf_dataset_specs
from .image_io import (
    build_image_io_settings,
    collect_image_sources,
    load_image,
    normalize_image_source,
)
from .image_augmentation import build_image_augmenter


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
        data_column_name: str,
        role_column_name: str,
        content_column_name: str,
        options_column_name: str,
        target_column_name: str,
        task_type_column_name: str,
        score_levels_column_name: str,
        id_column_name: str,
        pretrained_model_name: str,
        is_preprocessed: bool,
        custom_data_encoder_path: str,
        revision: str,
        reference_data_encoder_name: str,
        modality: str,
        left_padding: bool,
        is_enable_thinking: bool,
        max_length: int,
        padding_strategy: str,
        pad_to_multiple_of: Optional[int],
        candidate_markers: List[str],
        answer_prefix: str,
        option_template: str,
        max_pixels: Optional[int],
        do_resize: bool,
        image_augmentation: Dict[str, Any],
        dataset_image: Optional[Dict[str, Any]],
        dataset_subdir: Optional[str],
        dataset_file_path: Optional[str],
        dataset_file_paths: Optional[List[str]],
        dataset_files: Optional[List[Dict[str, Any]]],
        allow_dataset_file_name_mismatch: bool,
        val_dataset_file_path: Optional[str],
        val_dataset_file_paths: Optional[List[str]],
        val_dataset_files: Optional[List[Dict[str, Any]]],
        allow_val_dataset_file_name_mismatch: bool,
        dataset_resampling: Dict[str, Any],
    ) -> None:
        self.split = split
        self.data_column_name = data_column_name
        self.role_column_name = role_column_name
        self.content_column_name = content_column_name
        self.options_column_name = options_column_name
        self.target_column_name = target_column_name
        self.task_type_column_name = task_type_column_name
        self.score_levels_column_name = score_levels_column_name
        self.id_column_name = id_column_name
        self.modality = modality
        self.is_enable_thinking = is_enable_thinking
        self.max_length = max_length
        self.padding_strategy = padding_strategy
        self.candidate_markers = list(candidate_markers)
        self.answer_prefix = answer_prefix
        self.option_template = option_template
        self.max_pixels = max_pixels
        self.do_resize = do_resize
        if split not in {"train", "val", "test"}:
            raise ValueError("Decision split must be train, val, or test.")
        if padding_strategy not in {"dynamic", "max_length"}:
            raise ValueError("decision.padding_strategy must be dynamic or max_length.")
        if max_length <= 0 or not answer_prefix.strip():
            raise ValueError("Decision requires positive max_length and answer_prefix.")
        if max_pixels is not None and max_pixels <= 0:
            raise ValueError("max_pixels must be positive or null.")
        if pad_to_multiple_of is not None:
            if pad_to_multiple_of <= 0 or max_length % pad_to_multiple_of:
                raise ValueError("Decision max_length must satisfy pad_to_multiple_of.")
        self.image_settings = build_image_io_settings(
            dataset_image=dataset_image,
            default_image_root_dir=data_path,
        )
        self.image_augmenter = (
            build_image_augmenter(
                config=image_augmentation,
                seed=seed,
            )
            if split == "train"
            else None
        )
        encoder_path = (
            custom_data_encoder_path if is_preprocessed else pretrained_model_name
        )
        encoder_class = AutoTokenizer if modality == "text" else AutoProcessor
        self.data_encoder = encoder_class.from_pretrained(
            encoder_path,
            revision=revision,
        )
        tokenizer = (
            self.data_encoder if modality == "text" else self.data_encoder.tokenizer
        )
        if tokenizer.chat_template is None:
            reference = AutoTokenizer.from_pretrained(reference_data_encoder_name)
            tokenizer.chat_template = reference.chat_template
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.pad_token_id is None:
            raise ValueError("Decision padding requires a tokenizer pad or EOS token.")
        tokenizer.padding_side = "left" if left_padding else "right"
        self.tokenizer = tokenizer
        self.candidate_token_ids = build_decision_candidate_token_ids(
            data_encoder=self.data_encoder,
            candidate_markers=self.candidate_markers,
        )
        file_specs = resolve_dataset_file_specs(
            dataset_name=dataset_name,
            dataset_format=dataset_format,
            data_path=data_path,
            dataset_subdir=dataset_subdir,
            dataset_file_path=dataset_file_path,
            dataset_file_paths=dataset_file_paths,
            dataset_files=dataset_files,
            allow_dataset_file_name_mismatch=allow_dataset_file_name_mismatch,
            path_label="test_dataset" if split == "test" else "dataset",
            allow_weight=split != "test",
        )
        val_specs = resolve_optional_dataset_file_specs(
            dataset_name=dataset_name,
            dataset_format=dataset_format,
            data_path=data_path,
            dataset_file_path=val_dataset_file_path,
            dataset_file_paths=val_dataset_file_paths,
            dataset_files=val_dataset_files,
            allow_dataset_file_name_mismatch=allow_val_dataset_file_name_mismatch,
            path_label="val_dataset",
            allow_weight=False,
        )
        if split == "val" and val_specs is not None:
            self.rows = load_hf_dataset_specs(dataset_file_specs=val_specs)
        else:
            if dataset_resampling.enabled and split != "test":
                self.rows = load_weighted_hf_dataset_specs(
                    dataset_file_specs=file_specs,
                    dataset_resampling=dataset_resampling,
                    seed=seed,
                )
            else:
                self.rows = load_hf_dataset_specs(dataset_file_specs=file_specs)
            if val_specs is None and split != "test":
                partition = self.rows.train_test_split(
                    test_size=split_ratio,
                    seed=seed,
                )
                if split == "val":
                    self.rows = partition["test"]
                elif is_strict_split:
                    self.rows = partition["train"]
        if not len(self.rows):
            raise ValueError("Decision dataset must not be empty.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        row = self.rows[idx]
        parsed = self._parse_row(row=row)
        messages = self._build_messages(row=row)
        images = self._prepare_images(
            row=row,
            messages=messages,
        )
        options_text = "\n".join(
            self.option_template.format(
                marker=self.candidate_markers[index],
                text=option["text"],
            )
            for index, option in enumerate(parsed["options"])
        )
        content = messages[-1][self.content_column_name]
        if isinstance(content, str):
            messages[-1][self.content_column_name] = f"{content}\n{options_text}"
        else:
            content.append(
                {
                    "type": "text",
                    "text": f"\n{options_text}",
                }
            )
        text = self.data_encoder.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
            **build_enable_thinking_kwargs(
                data_encoder=self.data_encoder,
                is_enable_thinking=self.is_enable_thinking,
            ),
        )
        text += self.answer_prefix
        text_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        for marker, token_id in zip(
            self.candidate_markers[: len(parsed["options"])],
            self.candidate_token_ids,
        ):
            extended = self.tokenizer.encode(
                text + marker,
                add_special_tokens=False,
            )
            if extended != text_ids + [token_id]:
                raise ValueError(
                    "Decision candidate marker changes the prompt boundary."
                )
        kwargs = {
            "text": text,
            "padding": False,
            "truncation": False,
            "return_tensors": "pt",
        }
        if self.modality != "text":
            kwargs["images"] = images
        encoded = dict(self.data_encoder(**kwargs))
        for key in self.tokenizer.model_input_names + ["mm_token_type_ids"]:
            if key in encoded:
                encoded[key] = encoded[key].squeeze(0)
        if "pixel_values" in encoded and encoded["pixel_values"].dim() == 5:
            encoded["pixel_values"] = encoded["pixel_values"].squeeze(0)
        for key in [
            "pixel_attention_mask",
            "pixel_position_ids",
        ]:
            if key in encoded:
                encoded[key] = encoded[key].squeeze(0)
        if encoded["input_ids"].numel() > self.max_length:
            raise ValueError(
                f"Decision row {idx} exceeds max_length; truncation is disabled."
            )
        if encoded["input_ids"][-1].item() != text_ids[-1]:
            raise ValueError("Processor changed the Decision scoring suffix.")
        if self.padding_strategy == "max_length":
            for key in self.tokenizer.model_input_names + ["mm_token_type_ids"]:
                if key in encoded:
                    padding = self.max_length - encoded[key].size(0)
                    sides = (
                        (
                            padding,
                            0,
                        )
                        if self.tokenizer.padding_side == "left"
                        else (
                            0,
                            padding,
                        )
                    )
                    encoded[key] = torch.nn.functional.pad(
                        encoded[key],
                        sides,
                        value=self.tokenizer.pad_token_id if key == "input_ids" else 0,
                    )
        encoded["candidate_token_ids"] = torch.tensor(
            self.candidate_token_ids[: len(parsed["options"])],
            dtype=torch.long,
        )
        encoded["labels"] = torch.tensor(
            parsed["probabilities"],
            dtype=torch.float32,
        )
        encoded["decision_metadata"] = {
            "row_index": idx,
            "id": row[self.id_column_name],
            "task_type": parsed["task_type"],
            "candidate_ids": [option["id"] for option in parsed["options"]],
            "hard_label": parsed["hard_label"],
            "has_target": parsed["has_target"],
            "score_levels": parsed["score_levels"],
        }
        return encoded

    def to_hf_dataset(self) -> HFDataset:
        return HFDataset.from_dict({"_index": range(len(self))}).with_transform(
            _DecisionDatasetTransform(dataset=self)
        )

    def _parse_row(
        self,
        row: Dict[str, Any],
    ) -> Dict[str, Any]:
        options = row[self.options_column_name]
        if not isinstance(options, list) or not 2 <= len(options) <= len(
            self.candidate_markers
        ):
            raise ValueError(
                "Decision requires 2 or more options within marker capacity."
            )
        if any(
            not isinstance(option, dict)
            or not isinstance(option.get("id"), str)
            or not option["id"].strip()
            or not isinstance(option.get("text"), str)
            or not option["text"].strip()
            for option in options
        ):
            raise ValueError("Decision option id/text must be non-empty strings.")
        ids = [option["id"] for option in options]
        if len(set(ids)) != len(ids):
            raise ValueError("Decision option ids must be unique.")
        if (
            not isinstance(row.get(self.id_column_name), str)
            or not row[self.id_column_name].strip()
        ):
            raise ValueError("Decision row id must be a non-empty string.")
        task = row[self.task_type_column_name]
        levels = row.get(self.score_levels_column_name)
        if task == "noul":
            if len(ids) != 2 or set(ids) != {"true", "false"}:
                raise ValueError("Noul requires true and false candidate ids.")
        elif task == "score":
            if (
                not isinstance(levels, list)
                or len(levels) != len(ids)
                or any(not _is_finite_number(value=value) for value in levels)
                or any(left >= right for left, right in zip(levels, levels[1:]))
            ):
                raise ValueError("Score levels must be finite and strictly increasing.")
        elif task != "choice":
            raise ValueError("Decision task_type must be choice, noul, or score.")
        target = row.get(self.target_column_name)
        probabilities = [0.0] * len(ids)
        hard_label = -1
        if target is None:
            if self.split != "test":
                raise ValueError("Decision training and validation require targets.")
        else:
            if not isinstance(target, dict):
                raise ValueError(
                    "Decision target must be a label or distribution object."
                )
            target = {key: value for key, value in target.items() if value is not None}
            if set(target) == {"label_id"}:
                if target["label_id"] not in ids:
                    raise ValueError("Decision label_id must identify a valid option.")
                hard_label = ids.index(target["label_id"])
                probabilities[hard_label] = 1.0
            elif set(target) == {"probabilities"}:
                distribution = target["probabilities"]
                if isinstance(distribution, dict):
                    distribution = {
                        key: value
                        for key, value in distribution.items()
                        if value is not None
                    }
                if not isinstance(distribution, dict) or set(distribution) != set(ids):
                    raise ValueError(
                        "Decision distribution must cover exactly all option ids."
                    )
                probabilities = [distribution[option_id] for option_id in ids]
                if any(
                    not _is_finite_number(value=value) or not 0 <= value <= 1
                    for value in probabilities
                ) or not math.isclose(
                    sum(probabilities),
                    1.0,
                    abs_tol=1e-6,
                ):
                    raise ValueError(
                        "Decision probabilities must be finite, bounded, and sum to one."
                    )
            else:
                raise ValueError(
                    "Decision target requires exactly one label or distribution."
                )
        return {
            "options": options,
            "task_type": task,
            "score_levels": levels if task == "score" else None,
            "probabilities": probabilities,
            "hard_label": hard_label,
            "has_target": target is not None,
        }

    def _build_messages(
        self,
        row: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        content = row[self.data_column_name]
        if (
            not isinstance(content, (str, list))
            or not content
            or (isinstance(content, str) and not content.strip())
        ):
            raise ValueError(
                "Decision structural input must contain text or content parts."
            )
        return [
            {
                self.role_column_name: "user",
                self.content_column_name: copy.deepcopy(content),
            },
        ]

    def _prepare_images(
        self,
        row: Dict[str, Any],
        messages: List[Dict[str, Any]],
    ) -> Optional[List[Any]]:
        for message in messages:
            content = message[self.content_column_name]
            if isinstance(content, list):
                if not all(isinstance(part, dict) for part in content):
                    raise ValueError("Decision content parts must be objects.")
                message[self.content_column_name] = [
                    {key: value for key, value in part.items() if value is not None}
                    for part in content
                ]
        inline = collect_image_sources(value=messages)
        external = row.get("images")
        single = row.get("image")
        if external is not None and single is not None:
            raise ValueError("Decision images must use only one external image column.")
        if external is None and single is not None:
            external = [single]
        if external is not None and not isinstance(external, list):
            raise ValueError("Decision images column must be a list.")
        if self.modality == "text":
            if inline or external:
                raise ValueError(
                    "Decision text input must not silently discard images."
                )
            return None
        if external and any(source is not None for source in inline):
            raise ValueError(
                "Decision images must not be supplied both inline and externally."
            )
        sources = external if external else inline
        if not sources or any(source is None for source in sources):
            raise ValueError("Decision VLM input requires real image sources.")
        if inline and len(inline) != len(sources):
            raise ValueError(
                "Decision image placeholders and sources must have equal counts."
            )
        images = []
        for source in sources:
            normalized = normalize_image_source(
                image=source,
                **self.image_settings,
            )
            image = load_image(
                image=normalized,
                image_root_dir=self.image_settings["image_root_dir"],
                converted_image_mode=self.image_settings["converted_image_mode"],
            )
            if image is None:
                raise ValueError("Decision image could not be loaded.")
            if self.image_augmenter is not None:
                image = self.image_augmenter(image=image)
            if self.do_resize and self.max_pixels is not None:
                ratio = min(
                    1.0,
                    math.sqrt(self.max_pixels / (image.width * image.height)),
                )
                image = image.resize(
                    (
                        max(
                            1,
                            int(image.width * ratio),
                        ),
                        max(
                            1,
                            int(image.height * ratio),
                        ),
                    ),
                )
            images.append(image)
        if not inline:
            content = messages[-1][self.content_column_name]
            if isinstance(content, str):
                content = [
                    {
                        "type": "text",
                        "text": content,
                    },
                ]
            messages[-1][self.content_column_name] = [
                *[{"type": "image"} for _ in images],
                *content,
            ]
        return images


class ConversationalDataset(StructuralDataset):
    def _build_messages(
        self,
        row: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        messages = copy.deepcopy(row[self.data_column_name])
        if not isinstance(messages, list) or not messages:
            raise ValueError("Decision conversational input must be a message list.")
        if any(
            not isinstance(message, dict)
            or message.get(self.role_column_name) not in {"system", "user", "assistant"}
            or not isinstance(message.get(self.content_column_name), (str, list))
            or not message[self.content_column_name]
            for message in messages
        ):
            raise ValueError(
                "Decision messages require valid roles and non-empty content."
            )
        if messages[-1][self.role_column_name] != "user":
            raise ValueError(
                "Decision prompt must end with a user message, not a target."
            )
        return messages


class DecisionCollator:
    def __init__(
        self,
        model_collator: Callable[..., Dict[str, torch.Tensor]],
    ) -> None:
        self.model_collator = model_collator

    def __call__(
        self,
        batch: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Decision collator requires a non-empty batch.")
        output = self.model_collator(
            batch=[
                {
                    key: value
                    for key, value in sample.items()
                    if key not in {"candidate_token_ids", "labels", "decision_metadata"}
                }
                for sample in batch
            ],
        )
        capacity = max(len(sample["candidate_token_ids"]) for sample in batch)
        shape = (
            len(batch),
            capacity,
        )
        output["candidate_token_ids"] = torch.zeros(
            shape,
            dtype=torch.long,
        )
        output["candidate_mask"] = torch.zeros(
            shape,
            dtype=torch.bool,
        )
        output["labels"] = torch.zeros(shape)
        output["decision_metadata"] = [sample["decision_metadata"] for sample in batch]
        for index, sample in enumerate(batch):
            count = len(sample["candidate_token_ids"])
            output["candidate_token_ids"][index, :count] = sample["candidate_token_ids"]
            output["candidate_mask"][index, :count] = True
            output["labels"][index, :count] = sample["labels"]
        return output


def build_decision_candidate_token_ids(
    data_encoder: object,
    candidate_markers: List[str],
) -> List[int]:
    tokenizer = getattr(
        data_encoder,
        "tokenizer",
        data_encoder,
    )
    if len(candidate_markers) < 2:
        raise ValueError("Decision requires at least two candidate markers.")
    token_ids = []
    for marker in candidate_markers:
        if not isinstance(marker, str) or not marker.strip():
            raise ValueError("Decision candidate markers must be non-empty strings.")
        encoded = tokenizer.encode(
            marker,
            add_special_tokens=False,
        )
        if len(encoded) != 1 or encoded[0] in tokenizer.all_special_ids:
            raise ValueError(
                "Each Decision marker must encode to one non-special token."
            )
        token_ids.append(encoded[0])
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("Decision candidate markers must have unique token ids.")
    return token_ids


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(
            value,
            (
                int,
                float,
            ),
        )
        and not isinstance(
            value,
            bool,
        )
        and math.isfinite(value)
    )


class _DecisionDatasetTransform:
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
