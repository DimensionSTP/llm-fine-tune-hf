from typing import List, Tuple, Set, Any
import os
import json
import re
from urllib.parse import unquote_plus, urlsplit, urlunsplit

from omegaconf import DictConfig, ListConfig, OmegaConf


def redact_metadata_payload(
    config: DictConfig,
    payload: Any,
) -> Any:
    secret_values = _collect_metadata_secret_values(config=config)
    resolved_payload = _resolve_metadata_payload(payload=payload)
    return _redact_metadata_value(
        value=resolved_payload,
        secret_values=secret_values,
    )


def redact_metadata_text(
    config: DictConfig,
    text: str,
) -> str:
    try:
        secret_values = _collect_metadata_secret_values(config=config)
        return _redact_metadata_string(
            value=text,
            secret_values=secret_values,
        )
    except Exception:
        return "<redacted>"


def validate_metadata_file(
    config: DictConfig,
    path: str,
) -> None:
    payload, raw_text = _read_metadata_file(path=path)
    redacted_payload = redact_metadata_payload(
        config=config,
        payload=payload,
    )
    redacted_text = redact_metadata_text(
        config=config,
        text=raw_text,
    )
    if redacted_payload != payload or redacted_text != raw_text:
        raise ValueError("Metadata file contains sensitive information.")


def _resolve_metadata_payload(
    payload: Any,
) -> Any:
    if isinstance(payload, (DictConfig, ListConfig)):
        try:
            return OmegaConf.to_container(
                payload,
                resolve=True,
            )
        except Exception:
            raise ValueError("Metadata payload could not be resolved safely.") from None
    return payload


def _collect_metadata_secret_values(
    config: DictConfig,
) -> Set[str]:
    try:
        resolved_config = OmegaConf.to_container(
            config,
            resolve=True,
        )
    except Exception:
        raise ValueError("Metadata security context could not be resolved.") from None

    secret_values: Set[str] = set()
    _collect_secret_values_from_metadata(
        value=resolved_config,
        secret_values=secret_values,
    )
    secret_values.discard("")
    secret_values.discard("<redacted>")
    return secret_values


def _collect_secret_values_from_metadata(
    value: Any,
    secret_values: Set[str],
) -> None:
    if isinstance(value, dict):
        for item_key, item_value in value.items():
            if _is_sensitive_key(key=str(item_key)):
                _collect_sensitive_value_strings(
                    value=item_value,
                    secret_values=secret_values,
                )
            _collect_secret_values_from_metadata(
                value=item_value,
                secret_values=secret_values,
            )
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_secret_values_from_metadata(
                value=item,
                secret_values=secret_values,
            )
        return
    if isinstance(value, str):
        _collect_uri_secret_values(
            value=value,
            secret_values=secret_values,
        )


def _collect_sensitive_value_strings(
    value: Any,
    secret_values: Set[str],
) -> None:
    if isinstance(value, str):
        if value != "" and value != "<redacted>":
            secret_values.add(value)
        return
    if isinstance(value, dict):
        for item_value in value.values():
            _collect_sensitive_value_strings(
                value=item_value,
                secret_values=secret_values,
            )
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_sensitive_value_strings(
                value=item,
                secret_values=secret_values,
            )


def _collect_uri_secret_values(
    value: str,
    secret_values: Set[str],
) -> None:
    for match in re.finditer(
        _get_uri_pattern(),
        value,
    ):
        uri = match.group(0)
        try:
            parsed_uri = urlsplit(uri)
        except ValueError:
            continue
        userinfo, separator, _ = parsed_uri.netloc.rpartition("@")
        if separator:
            username, password_separator, password = userinfo.partition(":")
            if password_separator:
                _add_encoded_secret_value(
                    value=password,
                    secret_values=secret_values,
                )
            else:
                _add_encoded_secret_value(
                    value=username,
                    secret_values=secret_values,
                )
        for query_part in parsed_uri.query.split("&"):
            query_key, separator, query_value = query_part.partition("=")
            if separator and _is_sensitive_query_key(key=unquote_plus(query_key)):
                _add_encoded_secret_value(
                    value=query_value,
                    secret_values=secret_values,
                )


def _add_encoded_secret_value(
    value: str,
    secret_values: Set[str],
) -> None:
    if value == "":
        return
    secret_values.add(value)
    decoded_value = unquote_plus(value)
    if decoded_value != "":
        secret_values.add(decoded_value)


def _redact_metadata_value(
    value: Any,
    secret_values: Set[str],
) -> Any:
    if isinstance(value, dict):
        return {
            item_key: (
                _redact_sensitive_key_value(value=item_value)
                if _is_sensitive_key(key=str(item_key))
                else _redact_metadata_value(
                    value=item_value,
                    secret_values=secret_values,
                )
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [
            _redact_metadata_value(
                value=item,
                secret_values=secret_values,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _redact_metadata_value(
                value=item,
                secret_values=secret_values,
            )
            for item in value
        )
    if isinstance(value, str):
        return _redact_metadata_string(
            value=value,
            secret_values=secret_values,
        )
    return value


def _redact_sensitive_key_value(
    value: Any,
) -> Any:
    if value is None or value == "":
        return value
    if isinstance(value, (dict, list, tuple)) and len(value) == 0:
        return value
    return "<redacted>"


def _redact_metadata_string(
    value: str,
    secret_values: Set[str],
) -> str:
    redacted_value = re.sub(
        _get_sensitive_assignment_pattern(),
        _redact_sensitive_assignment,
        value,
    )
    redacted_value = re.sub(
        _get_uri_pattern(),
        _redact_uri_match,
        redacted_value,
    )
    for secret_value in sorted(
        secret_values,
        key=len,
        reverse=True,
    ):
        redacted_value = redacted_value.replace(
            secret_value,
            "<redacted>",
        )
    return redacted_value


def _redact_sensitive_assignment(
    match: re.Match[str],
) -> str:
    assignment_key = match.group("key")
    normalized_key = assignment_key.lstrip("+").rsplit(
        ".",
        1,
    )[-1]
    if not _is_sensitive_key(key=normalized_key):
        return match.group(0)
    return f"{assignment_key}=<redacted>"


def _redact_uri_match(
    match: re.Match[str],
) -> str:
    uri = match.group(0)
    try:
        parsed_uri = urlsplit(uri)
    except ValueError:
        return uri

    userinfo, separator, host = parsed_uri.netloc.rpartition("@")
    redacted_netloc = parsed_uri.netloc
    if separator and userinfo != "":
        redacted_netloc = f"<redacted>@{host}"
    redacted_query = _redact_uri_query(query=parsed_uri.query)
    return urlunsplit(
        (
            parsed_uri.scheme,
            redacted_netloc,
            parsed_uri.path,
            redacted_query,
            parsed_uri.fragment,
        )
    )


def _redact_uri_query(
    query: str,
) -> str:
    redacted_parts: List[str] = []
    for query_part in query.split("&"):
        query_key, separator, query_value = query_part.partition("=")
        if separator and _is_sensitive_query_key(key=unquote_plus(query_key)):
            redacted_parts.append(f"{query_key}=<redacted>")
        else:
            redacted_parts.append(query_part)
    return "&".join(redacted_parts)


def _is_sensitive_key(
    key: str,
) -> bool:
    normalized_key = key.lower()
    if normalized_key in _get_sensitive_exact_keys():
        return True
    return normalized_key.endswith(_get_sensitive_key_suffixes())


def _is_sensitive_query_key(
    key: str,
) -> bool:
    normalized_key = key.lower()
    if normalized_key == "signature" or normalized_key.endswith("_signature"):
        return True
    return _is_sensitive_key(key=normalized_key)


def _get_sensitive_exact_keys() -> Set[str]:
    return {
        "password",
        "secret",
        "api_key",
        "access_key",
        "private_key",
        "client_secret",
        "webhook_url",
        "authorization",
        "credential",
        "credentials",
        "auth_token",
        "access_token",
        "refresh_token",
        "bearer_token",
        "api_token",
        "hf_token",
        "hub_token",
        "session_token",
    }


def _get_sensitive_key_suffixes() -> Tuple[str, ...]:
    return (
        "_password",
        "_secret",
        "_api_key",
        "_access_key",
        "_private_key",
        "_client_secret",
        "_webhook_url",
        "_authorization",
        "_credential",
        "_credentials",
        "_auth_token",
        "_access_token",
        "_refresh_token",
        "_bearer_token",
        "_api_token",
        "_session_token",
    )


def _get_sensitive_assignment_pattern() -> str:
    return r"(?P<key>(?<!\S)\+{0,2}[A-Za-z_][A-Za-z0-9_.-]*)=" r"(?P<value>[^\s]+)"


def _get_uri_pattern() -> str:
    return r"[A-Za-z][A-Za-z0-9+.-]*://(?:<redacted>|[^\s\"'<>])+"


def _read_metadata_file(
    path: str,
) -> Tuple[Any, str]:
    try:
        with open(
            path,
            encoding="utf-8",
        ) as file:
            raw_text = file.read()
        extension = os.path.splitext(path)[1].lower()
        if extension == ".json":
            payload = json.loads(raw_text)
        elif extension in {".yaml", ".yml"}:
            parsed_config = OmegaConf.create(raw_text)
            payload = OmegaConf.to_container(
                parsed_config,
                resolve=False,
            )
        else:
            raise ValueError
    except Exception:
        raise ValueError("Metadata file could not be parsed safely.") from None
    return payload, raw_text
