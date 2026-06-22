from typing import Dict, List, Optional

import importlib

datasets = importlib.import_module("datasets")
HFDataset = datasets.Dataset
load_dataset = datasets.load_dataset

import pandas as pd


def load_pandas_dataset(
    dataset_format: str,
    dataset_file_paths: List[str],
) -> pd.DataFrame:
    frames = [
        load_pandas_dataset_file(
            dataset_format=dataset_format,
            dataset_file_path=dataset_file_path,
        )
        for dataset_file_path in dataset_file_paths
    ]
    if len(frames) == 1:
        return frames[0]
    return pd.concat(
        frames,
        ignore_index=True,
    )


def load_pandas_dataset_file(
    dataset_format: str,
    dataset_file_path: str,
) -> pd.DataFrame:
    if dataset_format == "parquet":
        return pd.read_parquet(dataset_file_path)
    if dataset_format in ["json", "jsonl"]:
        return pd.read_json(
            dataset_file_path,
            lines=True if dataset_format == "jsonl" else False,
        )
    if dataset_format in ["csv", "tsv"]:
        return pd.read_csv(
            dataset_file_path,
            sep="\t" if dataset_format == "tsv" else None,
        )
    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def load_hf_dataset(
    dataset_format: str,
    dataset_file_paths: List[str],
) -> HFDataset:
    hf_dataset_format = resolve_hf_dataset_format(dataset_format=dataset_format)
    return load_dataset(
        hf_dataset_format,
        data_files=dataset_file_paths,
    )["train"]


def load_hf_train_val_datasets(
    dataset_format: str,
    train_dataset_file_paths: List[str],
    val_dataset_file_paths: Optional[List[str]],
) -> Dict[str, Optional[HFDataset]]:
    train_dataset = load_hf_dataset(
        dataset_format=dataset_format,
        dataset_file_paths=train_dataset_file_paths,
    )
    val_dataset = (
        None
        if val_dataset_file_paths is None
        else load_hf_dataset(
            dataset_format=dataset_format,
            dataset_file_paths=val_dataset_file_paths,
        )
    )
    return {
        "train": train_dataset,
        "val": val_dataset,
    }


def resolve_hf_dataset_format(
    dataset_format: str,
) -> str:
    if dataset_format == "tsv":
        return "csv"
    if dataset_format == "jsonl":
        return "json"
    return dataset_format


__all__ = [
    "load_hf_dataset",
    "load_hf_train_val_datasets",
    "load_pandas_dataset",
    "load_pandas_dataset_file",
    "resolve_hf_dataset_format",
]
