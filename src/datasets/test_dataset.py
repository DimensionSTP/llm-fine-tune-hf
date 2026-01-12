from typing import Dict, List, Optional, Any
import os

import pandas as pd

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoProcessor


class StructuralDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        dataset_name: str,
        dataset_format: str,
        is_preprocessed: bool,
        data_column_name: str,
        target_column_name: str,
        role_column_name: str,
        content_column_name: str,
        assistant_column_name: str,
        pretrained_model_name: str,
        modality: str,
        custom_data_encoder_path: str,
        revision: str,
        reference_data_encoder_name: str,
        left_padding: bool,
        is_enable_thinking: bool,
        max_length: int,
    ) -> None:
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.is_preprocessed = is_preprocessed
        self.data_column_name = data_column_name
        self.target_column_name = target_column_name
        self.role_column_name = role_column_name
        self.content_column_name = content_column_name
        self.assistant_column_name = assistant_column_name
        self.pretrained_model_name = pretrained_model_name
        self.modality = modality

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

        self.is_enable_thinking = is_enable_thinking

        dataset = self.get_dataset()
        self.datas = dataset["datas"]
        self.labels = dataset["labels"]
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        applied = self.apply_chat_template(
            data=self.datas[idx],
            label=self.labels[idx],
        )
        prompt = applied["prompt"]
        label = applied["label"]

        image = None
        if self.modality != "text":
            image = []
            for part in self.datas[idx]:
                if isinstance(part, dict) and part.get("type") == "image":
                    image.append(part.get("image"))

            if not image:
                image = None

        encoded = self.encode_data(
            data=prompt,
            image=image,
        )
        if "token_type_ids" in encoded.keys():
            del encoded["token_type_ids"]
        if not "labels" in encoded.keys():
            encoded["labels"] = label
        return encoded

    def get_dataset(self) -> Dict[str, List[Any]]:
        file_name = f"{self.dataset_name}.{self.dataset_format}"
        full_data_path = os.path.join(
            self.data_path,
            file_name,
        )

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
        conversation = [
            {
                self.role_column_name: "user",
                self.content_column_name: (
                    data
                    if self.modality == "text"
                    else {
                        "type": "image",
                        "image": data,
                    }
                ),
            },
            {
                self.role_column_name: self.assistant_column_name,
                self.content_column_name: label,
            },
        ]

        label = conversation.pop()

        prompt = self.data_encoder.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.is_enable_thinking,
        )
        return {
            "prompt": prompt,
            "label": label,
        }

    def encode_data(
        self,
        data: str,
        image: Optional[List[str]],
    ) -> Dict[str, torch.Tensor]:
        kwargs = {
            "text": data,
            "padding": "max_length",
            "max_length": self.max_length,
            "truncation": True,
            "return_tensors": "pt",
            "add_special_tokens": True,
        }
        if self.modality != "text":
            kwargs["images"] = image
        encoded = self.data_encoder(**kwargs)

        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded


class ConversationalDataset(StructuralDataset):
    def __init__(
        self,
        data_path: str,
        dataset_name: str,
        dataset_format: str,
        is_preprocessed: bool,
        conversation_column_name: str,
        role_column_name: str,
        content_column_name: str,
        assistant_column_name: str,
        pretrained_model_name: str,
        modality: str,
        custom_data_encoder_path: str,
        revision: str,
        reference_data_encoder_name: str,
        left_padding: bool,
        is_enable_thinking: bool,
        max_length: int,
    ) -> None:
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.is_preprocessed = is_preprocessed
        self.conversation_column_name = conversation_column_name
        self.role_column_name = role_column_name
        self.content_column_name = content_column_name
        self.assistant_column_name = assistant_column_name
        self.pretrained_model_name = pretrained_model_name
        self.modality = modality

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

        self.is_enable_thinking = is_enable_thinking

        dataset = self.get_dataset()
        self.conversations = dataset["conversations"]
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        applied = self.apply_chat_template(
            conversation=self.conversations[idx],
        )
        prompt = applied["prompt"]
        label = applied["label"]

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

        encoded = self.encode_data(
            data=prompt,
            image=image,
        )
        if "token_type_ids" in encoded.keys():
            del encoded["token_type_ids"]
        if not "labels" in encoded.keys():
            encoded["labels"] = label
        return encoded

    def get_dataset(self) -> Dict[str, List[Any]]:
        file_name = f"{self.dataset_name}.{self.dataset_format}"
        full_data_path = os.path.join(
            self.data_path,
            file_name,
        )

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

        label = preprocessed_conversation.pop()[self.content_column_name]

        prompt = self.data_encoder.apply_chat_template(
            conversation=preprocessed_conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.is_enable_thinking,
        )
        return {
            "prompt": prompt,
            "label": label,
        }
