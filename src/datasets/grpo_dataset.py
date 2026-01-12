from typing import Dict, List
import os

import importlib

datasets = importlib.import_module("datasets")
HFDataset = datasets.Dataset
load_dataset = datasets.load_dataset


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

    def __call__(self) -> Dict[str, HFDataset]:
        file_name = f"{self.dataset_name}.{self.dataset_format}"
        full_data_path = os.path.join(
            self.data_path,
            file_name,
        )

        dataset = load_dataset(
            self.dataset_format,
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
                    label=examples[self.chosen_column_name][i],
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
        solution_column_name: str,
        reward_categories_column_name: str,
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

    def __call__(self) -> Dict[str, HFDataset]:
        file_name = f"{self.dataset_name}.{self.dataset_format}"
        full_data_path = os.path.join(
            self.data_path,
            file_name,
        )

        dataset = load_dataset(
            self.dataset_format,
            data_files=full_data_path,
        )["train"]

        if self.prompt_column_name != "prompt":
            dataset = dataset.rename_column(
                self.prompt_column_name,
                "prompt",
            )

        output_column_names = [
            "prompt",
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
