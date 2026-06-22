from typing import Dict, List, Optional, Any
import random

import importlib

datasets = importlib.import_module("datasets")
HFDataset = datasets.Dataset
concatenate_datasets = datasets.concatenate_datasets
load_dataset = datasets.load_dataset

import pandas as pd


def load_pandas_dataset(
    dataset_format: str,
    dataset_file_paths: List[str],
) -> pd.DataFrame:
    dataset_file_specs = [
        {
            "path": dataset_file_path,
            "format": dataset_format,
            "weight": None,
        }
        for dataset_file_path in dataset_file_paths
    ]
    return load_pandas_dataset_specs(dataset_file_specs=dataset_file_specs)


def load_pandas_dataset_specs(
    dataset_file_specs: List[Dict[str, Any]],
) -> pd.DataFrame:
    frames = [
        load_pandas_dataset_file(
            dataset_format=dataset_file_spec["format"],
            dataset_file_path=dataset_file_spec["path"],
        )
        for dataset_file_spec in dataset_file_specs
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
            sep="\t" if dataset_format == "tsv" else ",",
        )
    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def load_hf_dataset(
    dataset_format: str,
    dataset_file_paths: List[str],
) -> HFDataset:
    dataset_file_specs = [
        {
            "path": dataset_file_path,
            "format": dataset_format,
            "weight": None,
        }
        for dataset_file_path in dataset_file_paths
    ]
    return load_hf_dataset_specs(dataset_file_specs=dataset_file_specs)


def load_hf_dataset_specs(
    dataset_file_specs: List[Dict[str, Any]],
) -> HFDataset:
    datasets_by_spec = [
        load_hf_dataset_file(
            dataset_format=dataset_file_spec["format"],
            dataset_file_path=dataset_file_spec["path"],
        )
        for dataset_file_spec in dataset_file_specs
    ]
    if len(datasets_by_spec) == 1:
        return datasets_by_spec[0]
    return concatenate_datasets(
        dsets=_align_hf_dataset_features(datasets=datasets_by_spec)
    )


def load_hf_dataset_file(
    dataset_format: str,
    dataset_file_path: str,
) -> HFDataset:
    hf_dataset_format = resolve_hf_dataset_format(dataset_format=dataset_format)
    return load_dataset(
        hf_dataset_format,
        data_files=dataset_file_path,
    )["train"]


def load_hf_train_val_datasets(
    dataset_format: str,
    train_dataset_file_paths: List[str],
    val_dataset_file_paths: Optional[List[str]],
) -> Dict[str, Optional[HFDataset]]:
    train_dataset_specs = [
        {
            "path": train_dataset_file_path,
            "format": dataset_format,
            "weight": None,
        }
        for train_dataset_file_path in train_dataset_file_paths
    ]
    val_dataset_specs = (
        None
        if val_dataset_file_paths is None
        else [
            {
                "path": val_dataset_file_path,
                "format": dataset_format,
                "weight": None,
            }
            for val_dataset_file_path in val_dataset_file_paths
        ]
    )
    return load_hf_train_val_dataset_specs(
        train_dataset_file_specs=train_dataset_specs,
        val_dataset_file_specs=val_dataset_specs,
    )


def load_hf_train_val_dataset_specs(
    train_dataset_file_specs: List[Dict[str, Any]],
    val_dataset_file_specs: Optional[List[Dict[str, Any]]],
) -> Dict[str, Optional[HFDataset]]:
    train_dataset = load_hf_dataset_specs(
        dataset_file_specs=train_dataset_file_specs,
    )
    val_dataset = (
        None
        if val_dataset_file_specs is None
        else load_hf_dataset_specs(
            dataset_file_specs=val_dataset_file_specs,
        )
    )
    return {
        "train": train_dataset,
        "val": val_dataset,
    }


def load_weighted_pandas_dataset_specs(
    dataset_file_specs: List[Dict[str, Any]],
    dataset_resampling: Any,
    seed: int,
) -> pd.DataFrame:
    frames = [
        load_pandas_dataset_file(
            dataset_format=dataset_file_spec["format"],
            dataset_file_path=dataset_file_spec["path"],
        )
        for dataset_file_spec in dataset_file_specs
    ]
    sample_counts = build_weighted_sample_counts(
        source_lengths=[len(frame) for frame in frames],
        dataset_file_specs=dataset_file_specs,
        dataset_resampling=dataset_resampling,
    )
    sampled_frames = [
        frame.sample(
            n=sample_count,
            replace=dataset_resampling.replacement,
            random_state=seed + source_index,
        )
        for source_index, (frame, sample_count) in enumerate(
            zip(
                frames,
                sample_counts,
            )
        )
    ]
    return pd.concat(
        sampled_frames,
        ignore_index=True,
    )


def load_weighted_hf_dataset_specs(
    dataset_file_specs: List[Dict[str, Any]],
    dataset_resampling: Any,
    seed: int,
) -> HFDataset:
    hf_datasets = [
        load_hf_dataset_file(
            dataset_format=dataset_file_spec["format"],
            dataset_file_path=dataset_file_spec["path"],
        )
        for dataset_file_spec in dataset_file_specs
    ]
    sample_counts = build_weighted_sample_counts(
        source_lengths=[len(hf_dataset) for hf_dataset in hf_datasets],
        dataset_file_specs=dataset_file_specs,
        dataset_resampling=dataset_resampling,
    )
    sampled_datasets = [
        _sample_hf_dataset(
            dataset=hf_dataset,
            sample_count=sample_count,
            replacement=dataset_resampling.replacement,
            seed=seed + source_index,
        )
        for source_index, (hf_dataset, sample_count) in enumerate(
            zip(
                hf_datasets,
                sample_counts,
            )
        )
    ]
    if len(sampled_datasets) == 1:
        return sampled_datasets[0]
    return concatenate_datasets(
        dsets=_align_hf_dataset_features(datasets=sampled_datasets)
    )


def build_weighted_sample_counts(
    source_lengths: List[int],
    dataset_file_specs: List[Dict[str, Any]],
    dataset_resampling: Any,
) -> List[int]:
    weights = _get_dataset_file_weights(dataset_file_specs=dataset_file_specs)
    weight_sum = sum(weights)
    normalized_weights = [weight / weight_sum for weight in weights]
    target_size = _resolve_resampling_target_size(
        source_lengths=source_lengths,
        normalized_weights=normalized_weights,
        replacement=dataset_resampling.replacement,
        target_size=dataset_resampling.target_size,
    )
    sample_counts = _allocate_sample_counts(
        target_size=target_size,
        normalized_weights=normalized_weights,
    )
    _validate_sample_counts(
        source_lengths=source_lengths,
        sample_counts=sample_counts,
        replacement=dataset_resampling.replacement,
    )
    return sample_counts


def resolve_hf_dataset_format(
    dataset_format: str,
) -> str:
    if dataset_format == "tsv":
        return "csv"
    if dataset_format == "jsonl":
        return "json"
    return dataset_format


def _sample_hf_dataset(
    dataset: HFDataset,
    sample_count: int,
    replacement: bool,
    seed: int,
) -> HFDataset:
    if replacement:
        rng = random.Random(seed)
        indices = [rng.randrange(len(dataset)) for _ in range(sample_count)]
        return dataset.select(indices)
    return dataset.shuffle(seed=seed).select(range(sample_count))


def _align_hf_dataset_features(
    datasets: List[HFDataset],
) -> List[HFDataset]:
    reference_features = datasets[0].features
    return [dataset.cast(reference_features) for dataset in datasets]


def _get_dataset_file_weights(
    dataset_file_specs: List[Dict[str, Any]],
) -> List[float]:
    weights = [dataset_file_spec["weight"] for dataset_file_spec in dataset_file_specs]
    if any(weight is None for weight in weights):
        raise ValueError("dataset_files weight is required when resampling is enabled.")
    return weights


def _resolve_resampling_target_size(
    source_lengths: List[int],
    normalized_weights: List[float],
    replacement: bool,
    target_size: Optional[int],
) -> int:
    if target_size is not None:
        if target_size <= 0:
            raise ValueError("dataset_resampling.target_size must be positive.")
        return target_size
    if replacement:
        return sum(source_lengths)
    return min(
        int(source_length / normalized_weight)
        for source_length, normalized_weight in zip(
            source_lengths,
            normalized_weights,
        )
    )


def _allocate_sample_counts(
    target_size: int,
    normalized_weights: List[float],
) -> List[int]:
    raw_counts = [
        target_size * normalized_weight for normalized_weight in normalized_weights
    ]
    sample_counts = [int(raw_count) for raw_count in raw_counts]
    remainder = target_size - sum(sample_counts)
    remainder_order = sorted(
        range(len(raw_counts)),
        key=lambda index: raw_counts[index] - sample_counts[index],
        reverse=True,
    )
    for index in remainder_order[:remainder]:
        sample_counts[index] += 1
    if any(sample_count <= 0 for sample_count in sample_counts):
        raise ValueError("weighted resampling must sample at least one row per source.")
    return sample_counts


def _validate_sample_counts(
    source_lengths: List[int],
    sample_counts: List[int],
    replacement: bool,
) -> None:
    for source_length, sample_count in zip(
        source_lengths,
        sample_counts,
    ):
        if source_length <= 0:
            raise ValueError("dataset_files entries must not load empty datasets.")
        if not replacement and sample_count > source_length:
            raise ValueError(
                "weighted resampling requires replacement=true or a smaller target_size."
            )


__all__ = [
    "build_weighted_sample_counts",
    "load_hf_dataset",
    "load_hf_dataset_file",
    "load_hf_dataset_specs",
    "load_hf_train_val_datasets",
    "load_hf_train_val_dataset_specs",
    "load_weighted_hf_dataset_specs",
    "load_weighted_pandas_dataset_specs",
    "load_pandas_dataset",
    "load_pandas_dataset_file",
    "load_pandas_dataset_specs",
    "resolve_hf_dataset_format",
]
