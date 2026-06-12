from typing import Dict, List, Optional, Any
import os
import base64
import io
import urllib.request

from PIL import Image


def normalize_image_source(
    image: Any,
    image_root_dir: str,
    convert_unsupported_extensions: bool,
    unsupported_path_extensions: List[str],
    converted_image_mode: str,
) -> Any:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return image
    if isinstance(image, dict):
        return _normalize_image_dict(
            image=image,
            image_root_dir=image_root_dir,
            convert_unsupported_extensions=convert_unsupported_extensions,
            unsupported_path_extensions=unsupported_path_extensions,
            converted_image_mode=converted_image_mode,
        )
    if not isinstance(image, str):
        return image

    resolved_path = resolve_image_path(
        image_path=image,
        image_root_dir=image_root_dir,
    )
    if resolved_path is None:
        return image
    if not _path_exists(path=resolved_path):
        decoded_image = load_base64_image(
            value=image,
            converted_image_mode=converted_image_mode,
        )
        if decoded_image is not None:
            return decoded_image
    if _has_unsupported_extension(
        path=resolved_path,
        unsupported_path_extensions=unsupported_path_extensions,
    ):
        if not convert_unsupported_extensions:
            raise ValueError(
                f"Unsupported image extension for direct path input: {resolved_path}"
            )
        converted_image = load_image(
            image=resolved_path,
            image_root_dir=image_root_dir,
            converted_image_mode=converted_image_mode,
        )
        if converted_image is None:
            raise ValueError(
                f"Failed to decode unsupported image extension: {resolved_path}"
            )
        return converted_image
    return resolved_path


def normalize_image_payloads(
    value: Any,
    image_root_dir: str,
    convert_unsupported_extensions: bool,
    unsupported_path_extensions: List[str],
    converted_image_mode: str,
) -> Any:
    if isinstance(value, list):
        return [
            normalize_image_payloads(
                value=item,
                image_root_dir=image_root_dir,
                convert_unsupported_extensions=convert_unsupported_extensions,
                unsupported_path_extensions=unsupported_path_extensions,
                converted_image_mode=converted_image_mode,
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    normalized = dict(value)
    if normalized.get("type") == "image" and "image" in normalized:
        normalized["image"] = normalize_image_source(
            image=normalized["image"],
            image_root_dir=image_root_dir,
            convert_unsupported_extensions=convert_unsupported_extensions,
            unsupported_path_extensions=unsupported_path_extensions,
            converted_image_mode=converted_image_mode,
        )
    if "image" in normalized and normalized.get("type") != "image":
        normalized["image"] = normalize_image_payloads(
            value=normalized["image"],
            image_root_dir=image_root_dir,
            convert_unsupported_extensions=convert_unsupported_extensions,
            unsupported_path_extensions=unsupported_path_extensions,
            converted_image_mode=converted_image_mode,
        )
    if "images" in normalized:
        normalized["images"] = normalize_image_payloads(
            value=normalized["images"],
            image_root_dir=image_root_dir,
            convert_unsupported_extensions=convert_unsupported_extensions,
            unsupported_path_extensions=unsupported_path_extensions,
            converted_image_mode=converted_image_mode,
        )
    return normalized


def build_image_io_settings(
    dataset_image: Optional[Dict[str, Any]],
    default_image_root_dir: str,
) -> Dict[str, Any]:
    config = dataset_image or {}
    return {
        "image_root_dir": str(
            _select_config_value(
                config=config,
                key="image_root_dir",
                default=default_image_root_dir,
            )
        ),
        "convert_unsupported_extensions": bool(
            _select_config_value(
                config=config,
                key="convert_unsupported_extensions",
                default=True,
            )
        ),
        "unsupported_path_extensions": list(
            _select_config_value(
                config=config,
                key="unsupported_path_extensions",
                default=["tif", "tiff"],
            )
        ),
        "converted_image_mode": str(
            _select_config_value(
                config=config,
                key="converted_image_mode",
                default="RGB",
            )
        ),
    }


def load_image(
    image: Any,
    image_root_dir: str,
    converted_image_mode: Optional[str] = None,
) -> Optional[Image.Image]:
    if isinstance(image, Image.Image):
        return _convert_image_mode(
            image=image,
            converted_image_mode=converted_image_mode,
        )
    if isinstance(image, (bytes, bytearray)):
        return load_image_from_bytes(
            data=bytes(image),
            converted_image_mode=converted_image_mode,
        )
    if isinstance(image, dict):
        return _load_image_from_dict(
            image=image,
            image_root_dir=image_root_dir,
            converted_image_mode=converted_image_mode,
        )
    if not isinstance(image, str):
        return None

    value = image.strip()
    if value == "":
        return None

    if _is_url(value=value):
        try:
            with urllib.request.urlopen(value) as response:
                return load_image_from_bytes(
                    data=response.read(),
                    converted_image_mode=converted_image_mode,
                )
        except Exception:
            return None

    resolved_path = resolve_image_path(
        image_path=value,
        image_root_dir=image_root_dir,
    )
    if resolved_path is not None and os.path.exists(resolved_path):
        try:
            with open(resolved_path, "rb") as file:
                return load_image_from_bytes(
                    data=file.read(),
                    converted_image_mode=converted_image_mode,
                )
        except Exception:
            return None

    return load_base64_image(
        value=value,
        converted_image_mode=converted_image_mode,
    )


def load_image_from_bytes(
    data: bytes,
    converted_image_mode: Optional[str] = None,
) -> Optional[Image.Image]:
    try:
        image = Image.open(io.BytesIO(data))
    except Exception:
        return None
    return _convert_image_mode(
        image=image,
        converted_image_mode=converted_image_mode,
    )


def load_base64_image(
    value: str,
    converted_image_mode: Optional[str] = None,
) -> Optional[Image.Image]:
    try:
        if "base64," in value:
            _, value = value.split(
                "base64,",
                1,
            )
        decoded = base64.b64decode(
            value,
            validate=False,
        )
        return load_image_from_bytes(
            data=decoded,
            converted_image_mode=converted_image_mode,
        )
    except Exception:
        return None


def resolve_image_path(
    image_path: str,
    image_root_dir: str,
) -> Optional[str]:
    value = image_path.strip()
    if value == "":
        return None
    if _is_url(value=value):
        return None
    if value.startswith("data:") or "base64," in value:
        return None
    if not _looks_like_path(value=value):
        return None
    if os.path.isabs(value):
        return os.path.normpath(value)
    if not isinstance(image_root_dir, str) or image_root_dir.strip() == "":
        raise ValueError("image_root_dir must be a non-empty string.")
    return os.path.normpath(
        os.path.join(
            image_root_dir,
            value,
        )
    )


def _normalize_image_dict(
    image: Dict[str, Any],
    image_root_dir: str,
    convert_unsupported_extensions: bool,
    unsupported_path_extensions: List[str],
    converted_image_mode: str,
) -> Any:
    if image.get("bytes") is not None:
        return image

    path = image.get("path")
    if path is None:
        return image

    normalized_path = normalize_image_source(
        image=path,
        image_root_dir=image_root_dir,
        convert_unsupported_extensions=convert_unsupported_extensions,
        unsupported_path_extensions=unsupported_path_extensions,
        converted_image_mode=converted_image_mode,
    )
    if isinstance(normalized_path, Image.Image):
        return normalized_path

    normalized_image = dict(image)
    normalized_image["path"] = normalized_path
    return normalized_image


def _select_config_value(
    config: Dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    if key in config:
        return config[key]
    return default


def _load_image_from_dict(
    image: Dict[str, Any],
    image_root_dir: str,
    converted_image_mode: Optional[str],
) -> Optional[Image.Image]:
    data = image.get("bytes")
    if data is not None:
        return load_image_from_bytes(
            data=data,
            converted_image_mode=converted_image_mode,
        )

    path = image.get("path")
    if path is not None:
        return load_image(
            image=path,
            image_root_dir=image_root_dir,
            converted_image_mode=converted_image_mode,
        )

    return None


def _convert_image_mode(
    image: Image.Image,
    converted_image_mode: Optional[str],
) -> Image.Image:
    if converted_image_mode is None:
        return image
    if image.mode == converted_image_mode:
        return image
    return image.convert(converted_image_mode)


def _path_exists(
    path: str,
) -> bool:
    try:
        return os.path.exists(path)
    except OSError:
        return False


def _has_unsupported_extension(
    path: str,
    unsupported_path_extensions: List[str],
) -> bool:
    extension = os.path.splitext(path)[1].lstrip(".").lower()
    normalized_extensions = [
        item.lower().lstrip(".") for item in unsupported_path_extensions
    ]
    return extension in normalized_extensions


def _looks_like_path(
    value: str,
) -> bool:
    if "/" in value or "\\" in value:
        return True
    extension = os.path.splitext(value)[1].lower()
    return extension in {
        ".bmp",
        ".gif",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }


def _is_url(
    value: str,
) -> bool:
    return value.startswith(("http://", "https://"))
