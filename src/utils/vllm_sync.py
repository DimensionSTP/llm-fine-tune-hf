from typing import Dict, List, Tuple, Optional, Callable, Iterator, Any
import os
from contextlib import nullcontext
from fnmatch import fnmatchcase
from functools import partial
import hashlib
from importlib.metadata import version
import json
import shutil
import tempfile
from types import MethodType

from omegaconf import DictConfig, ListConfig

from packaging.specifiers import SpecifierSet

import torch

import bitsandbytes as bnb

import deepspeed

from safetensors import safe_open
from safetensors.torch import save_file


def prepare_vllm_server_accelerator_device(
    config: DictConfig,
) -> bool:
    world_size = os.environ.get(
        "WORLD_SIZE",
        "1",
    )
    should_prepare = (
        "use_vllm" in config
        and config.use_vllm
        and config.vllm_mode == "server"
        and world_size == "1"
        and torch.cuda.is_available()
    )
    if not should_prepare:
        return False

    configured_device = os.environ.get("ACCELERATE_TORCH_DEVICE")
    if configured_device is not None:
        device = torch.device(configured_device)
        if device.type != "cuda" or device.index is not None:
            return False

    os.environ["ACCELERATE_TORCH_DEVICE"] = f"cuda:{torch.cuda.current_device()}"
    return True


def resolve_vllm_lora_name_remap_config(
    config: DictConfig,
) -> Dict[str, Any]:
    remap_config = config.vllm_lora_name_remap
    runtime_package_versions = {
        str(package_name): version(str(package_name))
        for package_name in remap_config.version_packages
    }
    resolved = _resolve_vllm_lora_name_remap_profile(
        config=config,
        runtime_package_versions=runtime_package_versions,
    )
    remap_config.resolved_profile = resolved["profile"]
    remap_config.resolved_selector = resolved["selector"]
    remap_config.runtime_package_versions = runtime_package_versions
    return {
        "selection": str(remap_config.selection),
        "default_profile": str(remap_config.default_profile),
        "resolved_profile": resolved["profile"],
        "resolved_selector": resolved["selector"],
        "runtime_package_versions": runtime_package_versions,
        "prefix_rules": remap_config.profiles[resolved["profile"]].prefix_rules,
    }


def resolve_training_vllm_lora_name_remap_config(
    config: DictConfig,
) -> Dict[str, Any]:
    if config.fine_tune_method not in {
        "gold",
        "grpo",
        "async_grpo",
        "sdpo",
        "distillation",
    }:
        return {}
    if config.fine_tune_method != "async_grpo" and not config.use_vllm:
        return {}

    return resolve_vllm_lora_name_remap_config(
        config=config,
    )


def prepare_vllm_lora_request(
    config: DictConfig,
    lora_int_id: int,
) -> Dict[str, Any]:
    resolved_remap = resolve_vllm_lora_name_remap_config(
        config=config,
    )
    if not config.is_peft:
        return {
            "lora_request": None,
            "runtime": None,
        }

    source_adapter = _load_vllm_lora_source_adapter(
        adapter_path=str(config.peft_test.adapter_path),
    )
    prefix_rules = resolved_remap["prefix_rules"]
    if len(prefix_rules) == 0:
        effective_adapter_path = source_adapter["adapter_path"]
        action = "passthrough"
        remapped_tensor_count = 0
        passthrough_tensor_count = source_adapter["total_tensor_count"]
        effective_weights_sha256 = source_adapter["weights_sha256"]
    else:
        materialized_adapter = _materialize_vllm_lora_adapter(
            source_adapter=source_adapter,
            output_root=os.path.join(
                str(config.test_output_dir),
                "vllm_lora_adapters",
            ),
            resolved_profile=resolved_remap["resolved_profile"],
            prefix_rules=prefix_rules,
        )
        effective_adapter_path = materialized_adapter["adapter_path"]
        action = materialized_adapter["action"]
        remapped_tensor_count = materialized_adapter["remapped_tensor_count"]
        passthrough_tensor_count = materialized_adapter["passthrough_tensor_count"]
        effective_weights_sha256 = materialized_adapter["weights_sha256"]

    return {
        "lora_request": _build_vllm_lora_request(
            adapter_name=str(config.peft_test.adapter_name),
            lora_int_id=lora_int_id,
            adapter_path=effective_adapter_path,
        ),
        "runtime": {
            "source_weights_sha256": source_adapter["weights_sha256"],
            "source_config_sha256": source_adapter["config_sha256"],
            "effective_adapter_path": effective_adapter_path,
            "effective_weights_sha256": effective_weights_sha256,
            "total_tensor_count": source_adapter["total_tensor_count"],
            "remapped_tensor_count": remapped_tensor_count,
            "passthrough_tensor_count": passthrough_tensor_count,
            "action": action,
        },
    }


def patch_vllm_param_name_remap(
    trainer: Any,
    config: DictConfig,
) -> bool:
    if config.fine_tune_method == "async_grpo":
        return _patch_async_grpo_vllm_param_name_remap(
            trainer=trainer,
            config=config,
        )

    should_patch = (
        (
            config.fine_tune_method in {"gold", "distillation"}
            or (
                config.fine_tune_method == "grpo"
                and config.vllm_sync_strategy == "default"
            )
            or config.fine_tune_method == "sdpo"
        )
        and config.use_vllm
        and hasattr(
            trainer,
            "vllm_generation",
        )
        and not hasattr(
            trainer.vllm_generation,
            "_push_param_to_vllm_original",
        )
    )
    if not should_patch:
        return False

    resolve_training_vllm_lora_name_remap_config(config=config)
    trainer.vllm_generation._param_name_remapper = _get_lora_streaming_name_remapper(
        config=config,
    )
    trainer.vllm_generation._push_param_to_vllm_original = (
        trainer.vllm_generation._push_param_to_vllm
    )
    trainer.vllm_generation._push_param_to_vllm = MethodType(
        _push_param_to_vllm_with_name_remap,
        trainer.vllm_generation,
    )
    return True


def patch_qwen_packed_moe_vllm_sync(
    trainer: Any,
    config: DictConfig,
) -> bool:
    is_qwen_packed_moe = config.model_type.startswith("Qwen3-") and (
        "-experts_" in config.model_type
    )
    should_patch = (
        config.fine_tune_method == "grpo"
        and config.use_vllm
        and config.is_peft
        and config.dense_to_moe.router_with_lora
        and is_qwen_packed_moe
        and hasattr(
            trainer,
            "vllm_generation",
        )
    )
    if not should_patch:
        return False

    trainer.vllm_generation.sync_weights = MethodType(
        _build_router_with_lora_sync(
            remap_name=_remap_qwen_sparse_name,
        ),
        trainer.vllm_generation,
    )
    return True


def patch_sparse_decoder_moe_vllm_sync(
    trainer: Any,
    config: DictConfig,
) -> bool:
    is_sparse_decoder_moe = (
        "-experts_" in config.model_type and not config.model_type.startswith("Qwen3-")
    )
    should_patch = (
        config.fine_tune_method == "grpo"
        and config.use_vllm
        and config.is_peft
        and config.dense_to_moe.router_with_lora
        and is_sparse_decoder_moe
        and hasattr(
            trainer,
            "vllm_generation",
        )
    )
    if not should_patch:
        return False

    trainer.vllm_generation.sync_weights = MethodType(
        _build_router_with_lora_sync(
            remap_name=_remap_sparse_decoder_name,
        ),
        trainer.vllm_generation,
    )
    return True


def patch_lora_streaming_vllm_sync(
    trainer: Any,
    config: DictConfig,
) -> bool:
    should_patch = (
        config.fine_tune_method == "grpo"
        and config.use_vllm
        and config.is_peft
        and config.vllm_sync_strategy == "lora_streaming"
        and hasattr(
            trainer,
            "vllm_generation",
        )
    )
    if not should_patch:
        return False

    resolve_training_vllm_lora_name_remap_config(config=config)
    trainer.vllm_generation._lora_streaming_name_remapper = (
        _get_lora_streaming_name_remapper(config=config)
    )
    trainer.vllm_generation.sync_weights = MethodType(
        _sync_weights_lora_streaming,
        trainer.vllm_generation,
    )
    return True


def _load_vllm_lora_source_adapter(
    adapter_path: str,
) -> Dict[str, Any]:
    adapter_path = os.path.abspath(adapter_path)
    if not os.path.isdir(adapter_path):
        raise ValueError(f"PEFT adapter path must be a local directory: {adapter_path}")

    weights_path = os.path.join(
        adapter_path,
        "adapter_model.safetensors",
    )
    config_path = os.path.join(
        adapter_path,
        "adapter_config.json",
    )
    if not os.path.isfile(weights_path):
        adapter_files = os.listdir(adapter_path)
        has_unsupported_weights = "adapter_model.bin" in adapter_files or any(
            file_name == "adapter_model.safetensors.index.json"
            or (
                file_name.startswith("adapter_model-")
                and file_name.endswith(".safetensors")
            )
            for file_name in adapter_files
        )
        if has_unsupported_weights:
            raise ValueError(
                "Offline vLLM LoRA remap supports only adapter_model.safetensors."
            )
        raise FileNotFoundError(f"Missing PEFT adapter weights: {weights_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing PEFT adapter config: {config_path}")

    with open(
        config_path,
        encoding="utf-8",
    ) as file:
        adapter_config = json.load(file)
    if not isinstance(adapter_config, dict):
        raise ValueError("adapter_config.json must contain a JSON object.")

    tensors, metadata = _load_vllm_lora_tensors(weights_path=weights_path)
    return {
        "adapter_path": adapter_path,
        "weights_path": weights_path,
        "config_path": config_path,
        "weights_sha256": _hash_file(path=weights_path),
        "config_sha256": _hash_file(path=config_path),
        "tensors": tensors,
        "metadata": metadata,
        "total_tensor_count": len(tensors),
    }


def _materialize_vllm_lora_adapter(
    source_adapter: Dict[str, Any],
    output_root: str,
    resolved_profile: str,
    prefix_rules: ListConfig,
) -> Dict[str, Any]:
    output_root = os.path.abspath(output_root)
    if (
        resolved_profile in {".", ".."}
        or os.path.basename(resolved_profile) != resolved_profile
    ):
        raise ValueError("Resolved vLLM LoRA profile must be a path-safe name.")

    remapped_tensors, remapped_tensor_count = _remap_vllm_lora_adapter_tensors(
        tensors=source_adapter["tensors"],
        prefix_rules=prefix_rules,
    )
    if remapped_tensor_count == 0:
        raise ValueError(
            "Resolved vLLM LoRA profile did not remap any adapter tensors."
        )

    prefix_rule_signature = _build_vllm_lora_prefix_rule_signature(
        prefix_rules=prefix_rules,
    )
    identity = _build_vllm_lora_adapter_identity(
        source_weights_sha256=source_adapter["weights_sha256"],
        source_config_sha256=source_adapter["config_sha256"],
        resolved_profile=resolved_profile,
        prefix_rules=prefix_rules,
    )
    destination_path = os.path.join(
        output_root,
        f"{resolved_profile}-{identity}",
    )
    expected_manifest = {
        "source_weights_sha256": source_adapter["weights_sha256"],
        "source_config_sha256": source_adapter["config_sha256"],
        "resolved_profile": resolved_profile,
        "prefix_rule_signature": prefix_rule_signature,
        "total_tensor_count": source_adapter["total_tensor_count"],
        "remapped_tensor_count": remapped_tensor_count,
        "passthrough_tensor_count": (
            source_adapter["total_tensor_count"] - remapped_tensor_count
        ),
    }
    if os.path.exists(destination_path):
        validated = _validate_vllm_lora_artifact(
            artifact_path=destination_path,
            expected_manifest=expected_manifest,
            expected_tensors=remapped_tensors,
            expected_metadata=source_adapter["metadata"],
        )
        _validate_vllm_lora_source_hashes(source_adapter=source_adapter)
        return {
            "adapter_path": destination_path,
            "weights_sha256": validated["effective_weights_sha256"],
            "remapped_tensor_count": remapped_tensor_count,
            "passthrough_tensor_count": expected_manifest["passthrough_tensor_count"],
            "action": "reused",
        }

    os.makedirs(
        output_root,
        exist_ok=True,
    )
    temp_path = tempfile.mkdtemp(
        prefix=f".{resolved_profile}-{identity}.tmp.{os.getpid()}.",
        dir=output_root,
    )
    try:
        temp_weights_path = os.path.join(
            temp_path,
            "adapter_model.safetensors",
        )
        temp_config_path = os.path.join(
            temp_path,
            "adapter_config.json",
        )
        save_file(
            tensors=remapped_tensors,
            filename=temp_weights_path,
            metadata=source_adapter["metadata"],
        )
        _canonicalize_vllm_lora_safetensors(path=temp_weights_path)
        shutil.copyfile(
            source_adapter["config_path"],
            temp_config_path,
        )
        _validate_vllm_lora_source_hashes(source_adapter=source_adapter)
        effective_weights_sha256 = _hash_file(path=temp_weights_path)
        manifest = {
            **expected_manifest,
            "effective_weights_sha256": effective_weights_sha256,
        }
        _write_vllm_lora_manifest(
            manifest_path=os.path.join(
                temp_path,
                "remap_manifest.json",
            ),
            manifest=manifest,
        )
        _validate_vllm_lora_artifact(
            artifact_path=temp_path,
            expected_manifest=expected_manifest,
            expected_tensors=remapped_tensors,
            expected_metadata=source_adapter["metadata"],
        )
        try:
            os.rename(
                temp_path,
                destination_path,
            )
            action = "materialized"
        except OSError:
            if not os.path.exists(destination_path):
                raise
            validated = _validate_vllm_lora_artifact(
                artifact_path=destination_path,
                expected_manifest=expected_manifest,
                expected_tensors=remapped_tensors,
                expected_metadata=source_adapter["metadata"],
            )
            effective_weights_sha256 = validated["effective_weights_sha256"]
            action = "reused"
        _validate_vllm_lora_source_hashes(source_adapter=source_adapter)
        return {
            "adapter_path": destination_path,
            "weights_sha256": effective_weights_sha256,
            "remapped_tensor_count": remapped_tensor_count,
            "passthrough_tensor_count": expected_manifest["passthrough_tensor_count"],
            "action": action,
        }
    finally:
        if os.path.isdir(temp_path):
            shutil.rmtree(temp_path)


def _load_vllm_lora_tensors(
    weights_path: str,
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]:
    with safe_open(
        filename=weights_path,
        framework="pt",
        device="cpu",
    ) as file:
        tensors = {key: file.get_tensor(key) for key in sorted(file.keys())}
        metadata = file.metadata()
    return tensors, metadata


def _remap_vllm_lora_adapter_tensors(
    tensors: Dict[str, torch.Tensor],
    prefix_rules: ListConfig,
) -> Tuple[Dict[str, torch.Tensor], int]:
    remapped_tensors: Dict[str, torch.Tensor] = {}
    remapped_tensor_count = 0
    for source_key in sorted(tensors):
        target_key = _remap_vllm_lora_adapter_key(
            key=source_key,
            prefix_rules=prefix_rules,
        )
        if target_key in remapped_tensors:
            raise ValueError(f"vLLM LoRA remap key collision: {target_key}")
        remapped_tensors[target_key] = tensors[source_key]
        if target_key != source_key:
            remapped_tensor_count += 1
    return {
        key: remapped_tensors[key] for key in sorted(remapped_tensors)
    }, remapped_tensor_count


def _remap_vllm_lora_adapter_key(
    key: str,
    prefix_rules: ListConfig,
) -> str:
    wrapper = "base_model.model."
    if key.startswith(wrapper):
        model_name = key.removeprefix(wrapper)
        remapped_name = _remap_vllm_lora_name(
            name=model_name,
            prefix_rules=prefix_rules,
        )
        return f"{wrapper}{remapped_name}"
    return _remap_vllm_lora_name(
        name=key,
        prefix_rules=prefix_rules,
    )


def _build_vllm_lora_adapter_identity(
    source_weights_sha256: str,
    source_config_sha256: str,
    resolved_profile: str,
    prefix_rules: ListConfig,
) -> str:
    payload = {
        "source_weights_sha256": source_weights_sha256,
        "source_config_sha256": source_config_sha256,
        "resolved_profile": resolved_profile,
        "prefix_rules": _build_vllm_lora_prefix_rule_payload(
            prefix_rules=prefix_rules,
        ),
    }
    return _hash_bytes(
        value=_serialize_vllm_lora_identity(payload=payload),
    )


def _build_vllm_lora_prefix_rule_signature(
    prefix_rules: ListConfig,
) -> str:
    return _hash_bytes(
        value=_serialize_vllm_lora_identity(
            payload=_build_vllm_lora_prefix_rule_payload(
                prefix_rules=prefix_rules,
            ),
        ),
    )


def _build_vllm_lora_prefix_rule_payload(
    prefix_rules: ListConfig,
) -> List[Dict[str, str]]:
    return [
        {
            "source_prefix": str(rule.source_prefix),
            "target_prefix": str(rule.target_prefix),
        }
        for rule in prefix_rules
    ]


def _serialize_vllm_lora_identity(
    payload: Any,
) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _hash_bytes(
    value: bytes,
) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(
    path: str,
) -> str:
    digest = hashlib.sha256()
    with open(
        path,
        "rb",
    ) as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonicalize_vllm_lora_safetensors(
    path: str,
) -> None:
    with open(
        path,
        "rb",
    ) as file:
        payload = file.read()
    if len(payload) < 8:
        raise ValueError("Invalid vLLM LoRA safetensors payload.")

    header_length = int.from_bytes(
        payload[:8],
        byteorder="little",
    )
    data_start = 8 + header_length
    if data_start > len(payload):
        raise ValueError("Invalid vLLM LoRA safetensors header length.")
    header = json.loads(payload[8:data_start].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("Invalid vLLM LoRA safetensors header.")

    metadata = header.pop(
        "__metadata__",
        None,
    )
    canonical_header: Dict[str, Any] = {}
    if metadata is not None:
        canonical_header["__metadata__"] = {
            key: metadata[key] for key in sorted(metadata)
        }

    source_data = payload[data_start:]
    canonical_data = bytearray()
    for key in sorted(header):
        tensor_header = header[key]
        source_start, source_end = tensor_header["data_offsets"]
        target_start = len(canonical_data)
        canonical_data.extend(source_data[source_start:source_end])
        canonical_header[key] = {
            "dtype": tensor_header["dtype"],
            "shape": tensor_header["shape"],
            "data_offsets": [
                target_start,
                len(canonical_data),
            ],
        }

    header_bytes = json.dumps(
        canonical_header,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    canonical_payload = (
        len(header_bytes).to_bytes(
            8,
            byteorder="little",
        )
        + header_bytes
        + canonical_data
    )
    temp_path = f"{path}.canonical.{os.getpid()}"
    with open(
        temp_path,
        "wb",
    ) as file:
        file.write(canonical_payload)
    os.replace(
        temp_path,
        path,
    )


def _write_vllm_lora_manifest(
    manifest_path: str,
    manifest: Dict[str, Any],
) -> None:
    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def _validate_vllm_lora_source_hashes(
    source_adapter: Dict[str, Any],
) -> None:
    if (
        _hash_file(path=source_adapter["weights_path"])
        != source_adapter["weights_sha256"]
    ):
        raise ValueError("PEFT adapter weights changed during vLLM LoRA preparation.")
    if (
        _hash_file(path=source_adapter["config_path"])
        != source_adapter["config_sha256"]
    ):
        raise ValueError("PEFT adapter config changed during vLLM LoRA preparation.")


def _validate_vllm_lora_artifact(
    artifact_path: str,
    expected_manifest: Dict[str, Any],
    expected_tensors: Dict[str, torch.Tensor],
    expected_metadata: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    if not os.path.isdir(artifact_path):
        raise ValueError(f"Invalid vLLM LoRA artifact directory: {artifact_path}")
    expected_file_names = {
        "adapter_model.safetensors",
        "adapter_config.json",
        "remap_manifest.json",
    }
    if set(os.listdir(artifact_path)) != expected_file_names:
        raise ValueError(f"Incomplete vLLM LoRA artifact: {artifact_path}")

    weights_path = os.path.join(
        artifact_path,
        "adapter_model.safetensors",
    )
    config_path = os.path.join(
        artifact_path,
        "adapter_config.json",
    )
    manifest_path = os.path.join(
        artifact_path,
        "remap_manifest.json",
    )
    with open(
        manifest_path,
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError("remap_manifest.json must contain a JSON object.")
    if set(manifest) != set(expected_manifest) | {"effective_weights_sha256"}:
        raise ValueError("vLLM LoRA artifact manifest fields do not match.")
    if any(manifest[key] != value for key, value in expected_manifest.items()):
        raise ValueError("vLLM LoRA artifact manifest does not match its source.")
    if _hash_file(path=config_path) != expected_manifest["source_config_sha256"]:
        raise ValueError("vLLM LoRA artifact config hash does not match its source.")
    if _hash_file(path=weights_path) != manifest["effective_weights_sha256"]:
        raise ValueError("vLLM LoRA artifact weights hash does not match its manifest.")

    tensors, metadata = _load_vllm_lora_tensors(weights_path=weights_path)
    if list(tensors) != list(expected_tensors):
        raise ValueError("vLLM LoRA artifact tensor keys do not match.")
    if metadata != expected_metadata:
        raise ValueError("vLLM LoRA artifact metadata does not match.")
    for key, expected_tensor in expected_tensors.items():
        tensor = tensors[key]
        if (
            tensor.shape != expected_tensor.shape
            or tensor.dtype != expected_tensor.dtype
            or not torch.equal(
                tensor,
                expected_tensor,
            )
        ):
            raise ValueError(f"vLLM LoRA artifact tensor does not match: {key}")
    return manifest


def _build_vllm_lora_request(
    adapter_name: str,
    lora_int_id: int,
    adapter_path: str,
) -> Any:
    from vllm.lora.request import LoRARequest

    return LoRARequest(
        lora_name=adapter_name,
        lora_int_id=lora_int_id,
        lora_path=adapter_path,
    )


def _patch_async_grpo_vllm_param_name_remap(
    trainer: Any,
    config: DictConfig,
) -> bool:
    if trainer.weight_transfer is None:
        return False
    if hasattr(trainer, "_streaming_iter_original"):
        return False

    resolve_training_vllm_lora_name_remap_config(config=config)
    remap_name = _get_lora_streaming_name_remapper(config=config)
    weight_update_info = trainer.weight_transfer._weight_update_info
    weight_update_info["names"] = [
        remap_name(name) for name in weight_update_info["names"]
    ]
    trainer._vllm_param_name_remapper = remap_name
    trainer._streaming_iter_original = trainer._streaming_iter
    trainer._streaming_iter = MethodType(
        _streaming_iter_with_name_remap,
        trainer,
    )
    return True


def _streaming_iter_with_name_remap(
    self: Any,
) -> Iterator[Tuple[str, torch.Tensor]]:
    for name, parameter in self._streaming_iter_original():
        yield self._vllm_param_name_remapper(name), parameter


def _build_router_with_lora_sync(
    *,
    remap_name: Callable[[str], str],
) -> Callable[[Any], None]:
    return partial(
        _sync_weights_router_with_lora,
        remap_name=remap_name,
    )


def _push_param_to_vllm_with_name_remap(
    self: Any,
    name: str,
    param: torch.Tensor,
) -> None:
    remapped_name = self._param_name_remapper(name)
    self._push_param_to_vllm_original(
        remapped_name,
        param,
    )


def _sync_weights_router_with_lora(
    self: Any,
    *,
    remap_name: Callable[[str], str],
) -> None:
    if self.mode == "colocate" and self.enable_sleep_mode:
        torch.cuda.empty_cache()
        self.llm.wake_up(tags=["weights"])

    model = self.model
    accelerator = self.accelerator
    gather_if_zero3 = _get_gather_context(accelerator=accelerator)

    with gather_if_zero3(list(model.parameters())):
        model.merge_adapter()

        for name, param in model.named_parameters():
            name = name.removeprefix("base_model.model.").replace(
                ".base_layer",
                "",
            )

            if model.prefix in name:
                continue

            if "original_module" in name:
                continue

            name = self._fix_param_name_to_vllm(
                name,
                extra_prefixes=["modules_to_save.default."],
            )
            name = remap_name(name)

            if not name.endswith(".weight"):
                continue

            if self.mode == "server" and accelerator.is_main_process:
                self.vllm_client.update_named_param(
                    name,
                    param.data,
                )
            elif self.mode == "colocate":
                llm_model = (
                    self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                )
                llm_model.load_weights([(name, param.data)])

        model.unmerge_adapter()

    if self.mode == "server" and accelerator.is_main_process:
        self.vllm_client.reset_prefix_cache()
    elif self.mode == "colocate":
        self.llm.reset_prefix_cache()


def _get_gather_context(
    accelerator: Any,
) -> Callable:
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

    if zero_stage_3:
        return deepspeed.zero.GatheredParameters
    return nullcontext


def _is_lora_linear_module(
    module: Any,
) -> bool:
    return (
        hasattr(
            module,
            "base_layer",
        )
        and hasattr(
            module,
            "lora_A",
        )
        and hasattr(
            module,
            "lora_B",
        )
        and hasattr(
            module,
            "get_delta_weight",
        )
    )


def _get_active_lora_adapters(
    module: Any,
) -> List[str]:
    if getattr(module, "disable_adapters", False):
        return []

    active_adapters = getattr(
        module,
        "active_adapters",
        None,
    )
    if active_adapters is None:
        active_adapters = getattr(
            module,
            "active_adapter",
            "default",
        )
    if isinstance(active_adapters, str):
        active_adapters = [active_adapters]

    return [
        adapter
        for adapter in active_adapters
        if adapter in module.lora_A and adapter in module.lora_B
    ]


def _get_lora_sync_parameters(
    module: Any,
    adapters: List[str],
) -> List[torch.nn.Parameter]:
    parameters = [module.base_layer.weight]
    for adapter in adapters:
        parameters.append(module.lora_A[adapter].weight)
        parameters.append(module.lora_B[adapter].weight)
    return parameters


def _get_dense_base_weight(
    module: Any,
) -> torch.Tensor:
    base_weight = module.base_layer.weight
    quant_state = getattr(
        base_weight,
        "quant_state",
        None,
    )
    if quant_state is not None:
        return bnb.functional.dequantize_4bit(
            base_weight.data,
            quant_state=quant_state,
        )
    return base_weight.data


def _build_merged_lora_weight(
    module: Any,
    adapters: List[str],
) -> torch.Tensor:
    base_weight = _get_dense_base_weight(module=module)
    merged_weight = base_weight.clone()
    for adapter in adapters:
        delta_weight = module.get_delta_weight(adapter).to(
            device=base_weight.device,
            dtype=base_weight.dtype,
        )
        merged_weight.add_(delta_weight)
    return merged_weight


def _get_vllm_lora_weight_name(
    syncer: Any,
    module_name: str,
) -> str:
    name = module_name.removeprefix("base_model.model.")
    name = f"{name}.weight"
    return syncer._fix_param_name_to_vllm(
        name,
        extra_prefixes=["modules_to_save.default."],
    )


def _remap_vllm_lora_name(
    name: str,
    prefix_rules: ListConfig,
) -> str:
    matched_rules = [
        rule for rule in prefix_rules if name.startswith(rule.source_prefix)
    ]
    if len(matched_rules) == 0:
        return name
    if len(matched_rules) > 1:
        raise ValueError(
            f"Multiple vLLM LoRA name remap rules matched parameter: {name}"
        )

    rule = matched_rules[0]
    return f"{rule.target_prefix}{name.removeprefix(rule.source_prefix)}"


def _get_lora_streaming_name_remapper(
    config: DictConfig,
) -> Callable[[str], str]:
    profile = config.vllm_lora_name_remap.profiles[
        config.vllm_lora_name_remap.resolved_profile
    ]
    return partial(
        _remap_vllm_lora_name,
        prefix_rules=profile.prefix_rules,
    )


def _selector_matches_vllm_lora_runtime(
    selector: DictConfig,
    config: DictConfig,
    runtime_package_versions: Dict[str, str],
) -> bool:
    if len(selector.modalities) > 0 and config.modality not in selector.modalities:
        return False

    model_identifiers = [
        str(config.model_type),
        str(config.pretrained_model_name),
    ]
    if len(selector.model_patterns) > 0 and not any(
        fnmatchcase(model_identifier, pattern)
        for model_identifier in model_identifiers
        for pattern in selector.model_patterns
    ):
        return False

    return all(
        runtime_package_versions[package_name] in SpecifierSet(version_specifier)
        for package_name, version_specifier in selector.package_versions.items()
    )


def _resolve_vllm_lora_name_remap_profile(
    config: DictConfig,
    runtime_package_versions: Dict[str, str],
) -> Dict[str, str]:
    remap_config = config.vllm_lora_name_remap
    if remap_config.selection != "auto":
        return {
            "profile": str(remap_config.selection),
            "selector": "explicit",
        }

    matched_selectors = [
        selector
        for selector in remap_config.selectors
        if _selector_matches_vllm_lora_runtime(
            selector=selector,
            config=config,
            runtime_package_versions=runtime_package_versions,
        )
    ]
    if len(matched_selectors) > 1:
        selector_names = [str(selector.name) for selector in matched_selectors]
        raise ValueError(
            f"Multiple vLLM LoRA name remap selectors matched: {selector_names}"
        )
    if len(matched_selectors) == 1:
        selector = matched_selectors[0]
        return {
            "profile": str(selector.profile),
            "selector": str(selector.name),
        }

    return {
        "profile": str(remap_config.default_profile),
        "selector": "default",
    }


def _sync_weights_lora_streaming(
    self: Any,
) -> None:
    if self.mode == "colocate" and self.enable_sleep_mode:
        torch.cuda.empty_cache()
        self.llm.wake_up(tags=["weights"])

    model = self.model
    accelerator = self.accelerator
    gather_if_zero3 = _get_gather_context(accelerator=accelerator)
    remap_name = self._lora_streaming_name_remapper
    should_update = (self.mode == "server" and accelerator.is_main_process) or (
        self.mode == "colocate"
    )

    with torch.no_grad():
        for module_name, module in model.named_modules():
            if not _is_lora_linear_module(module=module):
                continue

            adapters = _get_active_lora_adapters(module=module)
            if len(adapters) == 0:
                continue

            parameters = _get_lora_sync_parameters(
                module=module,
                adapters=adapters,
            )
            with gather_if_zero3(parameters):
                if not should_update:
                    continue

                name = _get_vllm_lora_weight_name(
                    syncer=self,
                    module_name=module_name,
                )
                name = remap_name(name)
                merged_weight = _build_merged_lora_weight(
                    module=module,
                    adapters=adapters,
                )
                if self.mode == "server":
                    self.vllm_client.update_named_param(
                        name,
                        merged_weight,
                    )
                elif self.mode == "colocate":
                    llm_model = (
                        self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    )
                    llm_model.load_weights([(name, merged_weight)])

    if self.mode == "server" and accelerator.is_main_process:
        self.vllm_client.reset_prefix_cache()
    elif self.mode == "colocate":
        self.llm.reset_prefix_cache()


def _remap_qwen_sparse_name(
    name: str,
) -> str:
    if ".mlp.experts." in name:
        return ""

    is_attention_weight = ".self_attn." in name and name.endswith(".weight")
    is_router_weight = ".mlp.gate." in name and name.endswith(".weight")
    if is_attention_weight or is_router_weight:
        return name

    return ""


def _remap_sparse_decoder_name(
    name: str,
) -> str:
    if ".mlp.experts." in name:
        return ""

    if ".mlp.gate." in name:
        name = name.replace(
            ".mlp.gate.",
            ".block_sparse_moe.gate.",
        )

    is_attention_weight = ".self_attn." in name and name.endswith(".weight")
    is_router_weight = ".block_sparse_moe.gate." in name and name.endswith(".weight")
    if is_attention_weight or is_router_weight:
        return name

    return ""
