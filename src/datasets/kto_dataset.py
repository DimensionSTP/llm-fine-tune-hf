from typing import Dict, List, Optional

import importlib

datasets = importlib.import_module("datasets")
HFDataset = datasets.Dataset
load_dataset = datasets.load_dataset

from ..helpers.dataset_paths import resolve_dataset_file_path


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
        label_column_name: str,
        role_column_name: str,
        content_column_name: str,
        assistant_column_name: str,
        dataset_subdir: Optional[str] = None,
        dataset_file_path: Optional[str] = None,
        allow_dataset_file_name_mismatch: bool = False,
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
        self.target_column_name = target_column_name
        self.label_column_name = label_column_name
        self.role_column_name = role_column_name
        self.content_column_name = content_column_name
        self.assistant_column_name = assistant_column_name

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
            "completion",
            "label",
        ]
        input_column_names = [
            self.data_column_name,
            self.target_column_name,
            self.label_column_name,
        ]
        remove_columns = [
            name for name in input_column_names if name not in output_column_names
        ]

        dataset = dataset.map(
            self.create_conversations,
            batched=True,
            remove_columns=remove_columns,
        )

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
        prompts = []
        completions = []

        for i in range(len(examples[self.data_column_name])):
            prompts.append(
                self.apply_conversation_template(
                    data=examples[self.data_column_name][i],
                    is_prompt=True,
                )
            )
            completions.append(
                self.apply_conversation_template(
                    data=examples[self.target_column_name][i],
                    is_prompt=False,
                )
            )

        return {
            "prompt": prompts,
            "completion": completions,
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


class ConversationalDataset:
    def __init__(
        self,
        data_path: str,
        split_ratio: float,
        is_strict_split: bool,
        seed: int,
        dataset_name: str,
        dataset_format: str,
        prompt_column_name: str,
        completion_column_name: str,
        label_column_name: str,
        dataset_subdir: Optional[str] = None,
        dataset_file_path: Optional[str] = None,
        allow_dataset_file_name_mismatch: bool = False,
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
        self.completion_column_name = completion_column_name
        self.label_column_name = label_column_name

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

        if self.completion_column_name != "completion":
            dataset = dataset.rename_column(
                self.completion_column_name,
                "completion",
            )

        if self.label_column_name != "label":
            dataset = dataset.rename_column(
                self.label_column_name,
                "label",
            )

        output_column_names = [
            "prompt",
            "completion",
            "label",
        ]
        remove_columns = [
            column
            for column in dataset.column_names
            if column not in output_column_names
        ]

        if remove_columns:
            dataset = dataset.remove_columns(remove_columns)

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
