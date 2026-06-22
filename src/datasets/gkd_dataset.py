from typing import Dict, List, Optional

import importlib

datasets = importlib.import_module("datasets")
HFDataset = datasets.Dataset

from ..helpers.dataset_paths import (
    resolve_dataset_file_paths,
    resolve_optional_dataset_file_paths,
)
from .dataset_loading import load_hf_train_val_datasets


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
        target_column_name: str,
        role_column_name: str,
        content_column_name: str,
        assistant_column_name: str,
        dataset_subdir: Optional[str],
        dataset_file_path: Optional[str],
        dataset_file_paths: Optional[List[str]],
        allow_dataset_file_name_mismatch: bool,
        val_dataset_file_path: Optional[str],
        val_dataset_file_paths: Optional[List[str]],
        allow_val_dataset_file_name_mismatch: bool,
    ) -> None:
        self.data_path = data_path
        self.dataset_subdir = dataset_subdir
        self.dataset_file_path = dataset_file_path
        self.dataset_file_paths = dataset_file_paths
        self.allow_dataset_file_name_mismatch = allow_dataset_file_name_mismatch
        self.val_dataset_file_path = val_dataset_file_path
        self.val_dataset_file_paths = val_dataset_file_paths
        self.allow_val_dataset_file_name_mismatch = allow_val_dataset_file_name_mismatch
        self.split_ratio = split_ratio
        self.is_strict_split = is_strict_split
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.data_column_name = data_column_name
        self.target_column_name = target_column_name
        self.role_column_name = role_column_name
        self.content_column_name = content_column_name
        self.assistant_column_name = assistant_column_name

    def __call__(self) -> Dict[str, HFDataset]:
        train_data_paths = resolve_dataset_file_paths(
            dataset_name=self.dataset_name,
            dataset_format=self.dataset_format,
            data_path=self.data_path,
            dataset_subdir=self.dataset_subdir,
            dataset_file_path=self.dataset_file_path,
            dataset_file_paths=self.dataset_file_paths,
            allow_dataset_file_name_mismatch=self.allow_dataset_file_name_mismatch,
        )
        val_data_paths = resolve_optional_dataset_file_paths(
            dataset_name=self.dataset_name,
            dataset_format=self.dataset_format,
            data_path=self.data_path,
            dataset_file_path=self.val_dataset_file_path,
            dataset_file_paths=self.val_dataset_file_paths,
            allow_dataset_file_name_mismatch=self.allow_val_dataset_file_name_mismatch,
            path_label="val_dataset",
        )
        loaded_datasets = load_hf_train_val_datasets(
            dataset_format=self.dataset_format,
            train_dataset_file_paths=train_data_paths,
            val_dataset_file_paths=val_data_paths,
        )
        dataset = loaded_datasets["train"]
        val_dataset = loaded_datasets["val"]

        output_column_names = [
            "messages",
        ]
        input_column_names = [
            self.data_column_name,
            self.target_column_name,
        ]
        remove_columns = [
            name for name in input_column_names if name not in output_column_names
        ]

        dataset = dataset.map(
            self.create_conversations,
            batched=True,
            remove_columns=remove_columns,
        )
        if val_dataset is not None:
            val_dataset = val_dataset.map(
                self.create_conversations,
                batched=True,
                remove_columns=remove_columns,
            )

        if val_dataset is None:
            split_dataset = dataset.train_test_split(
                test_size=self.split_ratio,
                seed=self.seed,
            )
            if self.is_strict_split:
                train_dataset = split_dataset["train"]
            else:
                train_dataset = dataset
            val_dataset = split_dataset["test"]
        else:
            train_dataset = dataset

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
                    label=examples[self.target_column_name][i],
                )
            )

        return {
            "messages": conversations,
        }

    def apply_conversation_template(
        self,
        data: str,
        label: str,
    ) -> List[Dict[str, str]]:
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
        return conversation


class ConversationalDataset:
    def __init__(
        self,
        data_path: str,
        split_ratio: float,
        is_strict_split: bool,
        seed: int,
        dataset_name: str,
        dataset_format: str,
        conversation_column_name: str,
        dataset_subdir: Optional[str],
        dataset_file_path: Optional[str],
        dataset_file_paths: Optional[List[str]],
        allow_dataset_file_name_mismatch: bool,
        val_dataset_file_path: Optional[str],
        val_dataset_file_paths: Optional[List[str]],
        allow_val_dataset_file_name_mismatch: bool,
    ) -> None:
        self.data_path = data_path
        self.dataset_subdir = dataset_subdir
        self.dataset_file_path = dataset_file_path
        self.dataset_file_paths = dataset_file_paths
        self.allow_dataset_file_name_mismatch = allow_dataset_file_name_mismatch
        self.val_dataset_file_path = val_dataset_file_path
        self.val_dataset_file_paths = val_dataset_file_paths
        self.allow_val_dataset_file_name_mismatch = allow_val_dataset_file_name_mismatch
        self.split_ratio = split_ratio
        self.is_strict_split = is_strict_split
        self.seed = seed
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.conversation_column_name = conversation_column_name

    def __call__(self) -> Dict[str, HFDataset]:
        train_data_paths = resolve_dataset_file_paths(
            dataset_name=self.dataset_name,
            dataset_format=self.dataset_format,
            data_path=self.data_path,
            dataset_subdir=self.dataset_subdir,
            dataset_file_path=self.dataset_file_path,
            dataset_file_paths=self.dataset_file_paths,
            allow_dataset_file_name_mismatch=self.allow_dataset_file_name_mismatch,
        )
        val_data_paths = resolve_optional_dataset_file_paths(
            dataset_name=self.dataset_name,
            dataset_format=self.dataset_format,
            data_path=self.data_path,
            dataset_file_path=self.val_dataset_file_path,
            dataset_file_paths=self.val_dataset_file_paths,
            allow_dataset_file_name_mismatch=self.allow_val_dataset_file_name_mismatch,
            path_label="val_dataset",
        )
        loaded_datasets = load_hf_train_val_datasets(
            dataset_format=self.dataset_format,
            train_dataset_file_paths=train_data_paths,
            val_dataset_file_paths=val_data_paths,
        )
        dataset = loaded_datasets["train"]
        val_dataset = loaded_datasets["val"]

        if self.conversation_column_name != "messages":
            dataset = dataset.rename_column(
                self.conversation_column_name,
                "messages",
            )
            if val_dataset is not None:
                val_dataset = val_dataset.rename_column(
                    self.conversation_column_name,
                    "messages",
                )

        output_column_names = [
            "messages",
        ]
        remove_columns = [
            column
            for column in dataset.column_names
            if column not in output_column_names
        ]

        if remove_columns:
            dataset = dataset.remove_columns(remove_columns)
            if val_dataset is not None:
                val_dataset = val_dataset.remove_columns(remove_columns)

        if val_dataset is None:
            split_dataset = dataset.train_test_split(
                test_size=self.split_ratio,
                seed=self.seed,
            )
            if self.is_strict_split:
                train_dataset = split_dataset["train"]
            else:
                train_dataset = dataset
            val_dataset = split_dataset["test"]
        else:
            train_dataset = dataset

        return {
            "train": train_dataset,
            "val": val_dataset,
        }
