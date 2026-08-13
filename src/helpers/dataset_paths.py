from typing import Dict, List, Union, Optional, Any
import os

from omegaconf import DictConfig, ListConfig


def resolve_dataset_file_path(
    dataset_name: str,
    dataset_format: str,
    data_path: str,
    dataset_subdir: Optional[str],
    dataset_file_path: Optional[str],
    allow_dataset_file_name_mismatch: bool,
) -> str:
    expected_file_name = build_dataset_file_name(
        dataset_name=dataset_name,
        dataset_format=dataset_format,
    )
    normalized_data_path = _validate_data_path(data_path=data_path)
    normalized_dataset_file_path = _normalize_optional_path(
        path=dataset_file_path,
    )

    if normalized_dataset_file_path is not None:
        resolved_path = _resolve_dataset_file_override(
            data_path=normalized_data_path,
            dataset_file_path=normalized_dataset_file_path,
        )
        _validate_dataset_file_name(
            resolved_path=resolved_path,
            expected_file_name=expected_file_name,
            allow_dataset_file_name_mismatch=allow_dataset_file_name_mismatch,
        )
        return resolved_path

    normalized_dataset_subdir = _normalize_optional_path(path=dataset_subdir)
    if normalized_dataset_subdir is not None:
        if os.path.isabs(normalized_dataset_subdir):
            raise ValueError(
                "dataset_subdir must be relative. Override data_path for an absolute root."
            )
        return os.path.normpath(
            os.path.join(
                normalized_data_path,
                normalized_dataset_subdir,
                expected_file_name,
            )
        )

    return os.path.normpath(
        os.path.join(
            normalized_data_path,
            expected_file_name,
        )
    )


def resolve_dataset_file_paths(
    dataset_name: str,
    dataset_format: str,
    data_path: str,
    dataset_subdir: Optional[str],
    dataset_file_path: Optional[str],
    dataset_file_paths: Optional[Union[List[str], ListConfig]],
    allow_dataset_file_name_mismatch: bool,
) -> List[str]:
    normalized_dataset_file_paths = _normalize_optional_path_list(
        paths=dataset_file_paths,
    )
    if normalized_dataset_file_paths is None:
        return [
            resolve_dataset_file_path(
                dataset_name=dataset_name,
                dataset_format=dataset_format,
                data_path=data_path,
                dataset_subdir=dataset_subdir,
                dataset_file_path=dataset_file_path,
                allow_dataset_file_name_mismatch=allow_dataset_file_name_mismatch,
            )
        ]

    normalized_dataset_file_path = _normalize_optional_path(
        path=dataset_file_path,
    )
    if normalized_dataset_file_path is not None:
        raise ValueError(
            "dataset_file_path and dataset_file_paths are mutually exclusive."
        )

    expected_file_name = build_dataset_file_name(
        dataset_name=dataset_name,
        dataset_format=dataset_format,
    )
    normalized_data_path = _validate_data_path(data_path=data_path)
    resolved_paths = [
        _resolve_dataset_file_override(
            data_path=normalized_data_path,
            dataset_file_path=path,
        )
        for path in normalized_dataset_file_paths
    ]
    for resolved_path in resolved_paths:
        _validate_dataset_file_name(
            resolved_path=resolved_path,
            expected_file_name=expected_file_name,
            allow_dataset_file_name_mismatch=allow_dataset_file_name_mismatch,
        )
    return resolved_paths


def resolve_dataset_file_specs(
    dataset_name: str,
    dataset_format: str,
    data_path: str,
    dataset_subdir: Optional[str],
    dataset_file_path: Optional[str],
    dataset_file_paths: Optional[Union[List[str], ListConfig]],
    dataset_files: Optional[Union[List[Dict[str, Any]], ListConfig]],
    allow_dataset_file_name_mismatch: bool,
    path_label: str,
    allow_weight: bool,
) -> List[Dict[str, Any]]:
    normalized_dataset_files = _normalize_optional_dataset_file_specs(
        dataset_files=dataset_files,
        path_label=path_label,
        allow_weight=allow_weight,
    )
    if normalized_dataset_files is None:
        return [
            {
                "path": path,
                "format": dataset_format,
                "weight": None,
            }
            for path in resolve_dataset_file_paths(
                dataset_name=dataset_name,
                dataset_format=dataset_format,
                data_path=data_path,
                dataset_subdir=dataset_subdir,
                dataset_file_path=dataset_file_path,
                dataset_file_paths=dataset_file_paths,
                allow_dataset_file_name_mismatch=allow_dataset_file_name_mismatch,
            )
        ]

    if dataset_file_path is not None or dataset_file_paths is not None:
        raise ValueError(
            f"{path_label}_file_path(s) and {path_label}_files are mutually exclusive."
        )

    normalized_data_path = _validate_data_path(data_path=data_path)
    resolved_specs = [
        _resolve_dataset_file_spec(
            dataset_name=dataset_name,
            data_path=normalized_data_path,
            dataset_file_spec=dataset_file_spec,
            allow_dataset_file_name_mismatch=allow_dataset_file_name_mismatch,
        )
        for dataset_file_spec in normalized_dataset_files
    ]
    return resolved_specs


def resolve_optional_dataset_file_paths(
    dataset_name: str,
    dataset_format: str,
    data_path: str,
    dataset_file_path: Optional[str],
    dataset_file_paths: Optional[Union[List[str], ListConfig]],
    allow_dataset_file_name_mismatch: bool,
    path_label: str,
) -> Optional[List[str]]:
    normalized_dataset_file_paths = _normalize_optional_path_list(
        paths=dataset_file_paths,
    )
    normalized_dataset_file_path = _normalize_optional_path(
        path=dataset_file_path,
    )
    if (
        normalized_dataset_file_path is not None
        and normalized_dataset_file_paths is not None
    ):
        raise ValueError(
            f"{path_label}_file_path and {path_label}_file_paths are mutually exclusive."
        )
    if normalized_dataset_file_paths is None and normalized_dataset_file_path is None:
        return None

    expected_file_name = build_dataset_file_name(
        dataset_name=dataset_name,
        dataset_format=dataset_format,
    )
    normalized_data_path = _validate_data_path(data_path=data_path)
    source_paths = (
        [normalized_dataset_file_path]
        if normalized_dataset_file_paths is None
        else normalized_dataset_file_paths
    )
    resolved_paths = [
        _resolve_dataset_file_override(
            data_path=normalized_data_path,
            dataset_file_path=path,
        )
        for path in source_paths
    ]
    for resolved_path in resolved_paths:
        _validate_dataset_file_name(
            resolved_path=resolved_path,
            expected_file_name=expected_file_name,
            allow_dataset_file_name_mismatch=allow_dataset_file_name_mismatch,
        )
    return resolved_paths


def resolve_optional_dataset_file_specs(
    dataset_name: str,
    dataset_format: str,
    data_path: str,
    dataset_file_path: Optional[str],
    dataset_file_paths: Optional[Union[List[str], ListConfig]],
    dataset_files: Optional[Union[List[Dict[str, Any]], ListConfig]],
    allow_dataset_file_name_mismatch: bool,
    path_label: str,
    allow_weight: bool,
) -> Optional[List[Dict[str, Any]]]:
    normalized_dataset_files = _normalize_optional_dataset_file_specs(
        dataset_files=dataset_files,
        path_label=path_label,
        allow_weight=allow_weight,
    )
    if normalized_dataset_files is None:
        resolved_paths = resolve_optional_dataset_file_paths(
            dataset_name=dataset_name,
            dataset_format=dataset_format,
            data_path=data_path,
            dataset_file_path=dataset_file_path,
            dataset_file_paths=dataset_file_paths,
            allow_dataset_file_name_mismatch=allow_dataset_file_name_mismatch,
            path_label=path_label,
        )
        if resolved_paths is None:
            return None
        return [
            {
                "path": path,
                "format": dataset_format,
                "weight": None,
            }
            for path in resolved_paths
        ]

    if dataset_file_path is not None or dataset_file_paths is not None:
        raise ValueError(
            f"{path_label}_file_path(s) and {path_label}_files are mutually exclusive."
        )

    normalized_data_path = _validate_data_path(data_path=data_path)
    resolved_specs = [
        _resolve_dataset_file_spec(
            dataset_name=dataset_name,
            data_path=normalized_data_path,
            dataset_file_spec=dataset_file_spec,
            allow_dataset_file_name_mismatch=allow_dataset_file_name_mismatch,
        )
        for dataset_file_spec in normalized_dataset_files
    ]
    return resolved_specs


def build_dataset_file_path_metadata(
    dataset_name: str,
    dataset_format: str,
    data_path: str,
    dataset_subdir: Optional[str],
    dataset_file_path: Optional[str],
    allow_dataset_file_name_mismatch: bool,
) -> Dict[str, Any]:
    resolved_path = resolve_dataset_file_path(
        dataset_name=dataset_name,
        dataset_format=dataset_format,
        data_path=data_path,
        dataset_subdir=dataset_subdir,
        dataset_file_path=dataset_file_path,
        allow_dataset_file_name_mismatch=allow_dataset_file_name_mismatch,
    )
    expected_file_name = build_dataset_file_name(
        dataset_name=dataset_name,
        dataset_format=dataset_format,
    )
    return {
        "dataset_name": dataset_name,
        "dataset_format": dataset_format,
        "data_path": os.path.normpath(str(data_path)),
        "dataset_subdir": _normalize_optional_path(path=dataset_subdir),
        "dataset_file_path": _normalize_optional_path(path=dataset_file_path),
        "resolved_dataset_file_path": resolved_path,
        "expected_dataset_file_name": expected_file_name,
        "dataset_file_name_mismatch": os.path.basename(resolved_path)
        != expected_file_name,
        "allow_dataset_file_name_mismatch": allow_dataset_file_name_mismatch,
    }


def build_train_dataset_input_metadata(
    dataset_name: str,
    dataset_format: str,
    data_path: str,
    dataset_subdir: Optional[str],
    dataset_file_path: Optional[str],
    dataset_file_paths: Optional[Union[List[str], ListConfig]],
    dataset_files: Optional[Union[List[Dict[str, Any]], ListConfig]],
    allow_dataset_file_name_mismatch: bool,
    val_dataset_file_path: Optional[str],
    val_dataset_file_paths: Optional[Union[List[str], ListConfig]],
    val_dataset_files: Optional[Union[List[Dict[str, Any]], ListConfig]],
    allow_val_dataset_file_name_mismatch: bool,
    use_validation: bool,
    dataset_resampling: DictConfig,
) -> Dict[str, Any]:
    train_specs = resolve_dataset_file_specs(
        dataset_name=dataset_name,
        dataset_format=dataset_format,
        data_path=data_path,
        dataset_subdir=dataset_subdir,
        dataset_file_path=dataset_file_path,
        dataset_file_paths=dataset_file_paths,
        dataset_files=dataset_files,
        allow_dataset_file_name_mismatch=allow_dataset_file_name_mismatch,
        path_label="dataset",
        allow_weight=True,
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
    return {
        "train": {
            "mode": _get_train_input_mode(
                specs=train_specs,
                dataset_resampling=dataset_resampling,
            ),
            "files": train_specs,
        },
        "validation": {
            "mode": _get_validation_input_mode(
                specs=val_specs,
                use_validation=use_validation,
            ),
            "files": val_specs,
            "from_train_split": use_validation and val_specs is None,
        },
    }


def resolve_effective_dataset_name(
    dataset_name: str,
    dataset_mix_name: Optional[str],
    dataset_file_paths: Optional[Union[List[str], ListConfig]],
    dataset_files: Optional[Union[List[Dict[str, Any]], ListConfig]],
) -> str:
    normalized_dataset_file_paths = _normalize_optional_path_list(
        paths=dataset_file_paths,
    )
    normalized_dataset_files = _normalize_optional_dataset_file_specs(
        dataset_files=dataset_files,
        path_label="dataset",
        allow_weight=True,
    )
    if normalized_dataset_file_paths is None and normalized_dataset_files is None:
        return dataset_name

    normalized_dataset_mix_name = _normalize_optional_path(
        path=dataset_mix_name,
    )
    if normalized_dataset_mix_name is None:
        raise ValueError(
            "dataset_mix_name is required when dataset_file_paths or dataset_files is set."
        )
    return normalized_dataset_mix_name


def build_dataset_file_name(
    dataset_name: str,
    dataset_format: str,
) -> str:
    if not isinstance(dataset_name, str) or dataset_name.strip() == "":
        raise ValueError("dataset_name must be a non-empty string.")
    if not isinstance(dataset_format, str) or dataset_format.strip() == "":
        raise ValueError("dataset_format must be a non-empty string.")
    return f"{dataset_name}.{dataset_format}"


def _validate_data_path(
    data_path: str,
) -> str:
    if not isinstance(data_path, str) or data_path.strip() == "":
        raise ValueError("data_path must be a non-empty string.")
    return os.path.normpath(data_path)


def _normalize_optional_path(
    path: Optional[str],
) -> Optional[str]:
    if path is None:
        return None
    if not isinstance(path, str):
        raise ValueError("optional path values must be strings or null.")
    normalized = path.strip()
    if normalized == "":
        return None
    return normalized


def _normalize_optional_path_list(
    paths: Optional[Union[List[str], ListConfig]],
) -> Optional[List[str]]:
    if paths is None:
        return None
    if not isinstance(paths, (list, ListConfig)):
        raise ValueError("optional path list values must be lists or null.")
    normalized_paths = [_normalize_required_path(path=path) for path in paths]
    if len(normalized_paths) == 0:
        raise ValueError("optional path list values must not be empty.")
    return normalized_paths


def _normalize_optional_dataset_file_specs(
    dataset_files: Optional[Union[List[Dict[str, Any]], ListConfig]],
    path_label: str,
    allow_weight: bool,
) -> Optional[List[Dict[str, Any]]]:
    if dataset_files is None:
        return None
    if not isinstance(dataset_files, (list, ListConfig)):
        raise ValueError(f"{path_label}_files must be a list or null.")
    if len(dataset_files) == 0:
        raise ValueError(f"{path_label}_files must not be empty.")
    return [
        _normalize_dataset_file_spec(
            dataset_file_spec=dataset_file_spec,
            path_label=path_label,
            allow_weight=allow_weight,
        )
        for dataset_file_spec in dataset_files
    ]


def _normalize_dataset_file_spec(
    dataset_file_spec: Any,
    path_label: str,
    allow_weight: bool,
) -> Dict[str, Any]:
    if not isinstance(dataset_file_spec, (dict, DictConfig)):
        raise ValueError(f"{path_label}_files entries must be mappings.")
    for required_key in ["path", "format"]:
        if required_key not in dataset_file_spec:
            raise ValueError(f"{path_label}_files entries require {required_key}.")
    weight = None
    if "weight" in dataset_file_spec:
        if dataset_file_spec["weight"] is None:
            weight = None
        elif not allow_weight:
            raise ValueError(f"{path_label}_files do not support weight.")
        else:
            weight = _normalize_dataset_file_weight(
                weight=dataset_file_spec["weight"],
                path_label=path_label,
            )
    return {
        "path": _normalize_required_path(path=dataset_file_spec["path"]),
        "format": _normalize_required_path(path=dataset_file_spec["format"]),
        "weight": weight,
    }


def _normalize_dataset_file_weight(
    weight: Any,
    path_label: str,
) -> float:
    if not isinstance(weight, (int, float)):
        raise ValueError(f"{path_label}_files weight must be numeric.")
    if weight <= 0:
        raise ValueError(f"{path_label}_files weight must be positive.")
    return float(weight)


def _normalize_required_path(
    path: Any,
) -> str:
    if not isinstance(path, str):
        raise ValueError("path list values must be strings.")
    normalized = path.strip()
    if normalized == "":
        raise ValueError("path list values must be non-empty strings.")
    return normalized


def _resolve_dataset_file_spec(
    dataset_name: str,
    data_path: str,
    dataset_file_spec: Dict[str, Any],
    allow_dataset_file_name_mismatch: bool,
) -> Dict[str, Any]:
    expected_file_name = build_dataset_file_name(
        dataset_name=dataset_name,
        dataset_format=dataset_file_spec["format"],
    )
    resolved_path = _resolve_dataset_file_override(
        data_path=data_path,
        dataset_file_path=dataset_file_spec["path"],
    )
    _validate_dataset_file_name(
        resolved_path=resolved_path,
        expected_file_name=expected_file_name,
        allow_dataset_file_name_mismatch=allow_dataset_file_name_mismatch,
    )
    return {
        "path": resolved_path,
        "format": dataset_file_spec["format"],
        "weight": dataset_file_spec["weight"],
    }


def _resolve_dataset_file_override(
    data_path: str,
    dataset_file_path: str,
) -> str:
    if os.path.isabs(dataset_file_path):
        return os.path.normpath(dataset_file_path)
    return os.path.normpath(
        os.path.join(
            data_path,
            dataset_file_path,
        )
    )


def _validate_dataset_file_name(
    resolved_path: str,
    expected_file_name: str,
    allow_dataset_file_name_mismatch: bool,
) -> None:
    if os.path.basename(resolved_path) == expected_file_name:
        return
    if allow_dataset_file_name_mismatch:
        return
    raise ValueError(
        "dataset_file_path basename must match "
        f"{expected_file_name}. Set allow_dataset_file_name_mismatch=true to opt out."
    )


def _get_input_mode(
    paths: List[str],
) -> str:
    if len(paths) == 1:
        return "single_file"
    return "multi_file"


def _get_spec_input_mode(
    specs: List[Dict[str, Any]],
) -> str:
    if len(specs) == 1:
        return "single_file"
    formats = {spec["format"] for spec in specs}
    if len(formats) == 1:
        return "multi_file"
    return "multi_format"


def _get_train_input_mode(
    specs: List[Dict[str, Any]],
    dataset_resampling: DictConfig,
) -> str:
    if dataset_resampling.enabled:
        return "weighted_resampled"
    return _get_spec_input_mode(specs=specs)


def _get_validation_input_mode(
    specs: Optional[List[Dict[str, Any]]],
    use_validation: bool,
) -> str:
    if not use_validation:
        return "disabled"
    if specs is None:
        return "sampled_from_train"
    if len(specs) == 1:
        return "external_single_file"
    if _get_spec_input_mode(specs=specs) == "multi_format":
        return "external_multi_format"
    return "external_multi_file"


__all__ = [
    "build_dataset_file_name",
    "build_dataset_file_path_metadata",
    "build_train_dataset_input_metadata",
    "resolve_effective_dataset_name",
    "resolve_dataset_file_path",
    "resolve_dataset_file_paths",
    "resolve_dataset_file_specs",
    "resolve_optional_dataset_file_paths",
    "resolve_optional_dataset_file_specs",
]
