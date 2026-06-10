from typing import Dict, Optional, Any
import os


def resolve_dataset_file_path(
    dataset_name: str,
    dataset_format: str,
    data_path: str,
    dataset_subdir: Optional[str] = None,
    dataset_file_path: Optional[str] = None,
    allow_dataset_file_name_mismatch: bool = False,
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


def build_dataset_file_path_metadata(
    dataset_name: str,
    dataset_format: str,
    data_path: str,
    dataset_subdir: Optional[str] = None,
    dataset_file_path: Optional[str] = None,
    allow_dataset_file_name_mismatch: bool = False,
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


__all__ = [
    "build_dataset_file_name",
    "build_dataset_file_path_metadata",
    "resolve_dataset_file_path",
]
