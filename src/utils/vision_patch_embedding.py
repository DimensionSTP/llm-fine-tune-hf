from typing import Dict, List, Tuple, Set, Optional, Any
import os
import importlib.metadata
import json
from types import MethodType
import warnings

from omegaconf import DictConfig, ListConfig, OmegaConf

import torch
from torch import distributed as dist
from torch import nn
import torch.nn.functional as F

from .vision_patch_embedding_probe import run_vision_patch_embedding_probe


def validate_vision_patch_embedding_config(
    config: DictConfig,
) -> None:
    compatibility_config = config.vision_patch_embedding
    if compatibility_config.mode not in {"native", "linear", "auto"}:
        raise ValueError("vision_patch_embedding.mode must be native, linear, or auto.")
    if compatibility_config.mode == "linear" and config.modality == "text":
        raise ValueError(
            "vision_patch_embedding.mode=linear requires an image-text model."
        )
    if (
        config.modality != "text"
        and compatibility_config.mode != "native"
        and (
            (config.mode == "train" and config.fine_tune_method == "async_grpo")
            or config.mode in {"test_vllm", "test_vllm_multi_turn"}
        )
    ):
        raise ValueError(
            "Non-native vision patch embedding modes require a repository-owned "
            "Hugging Face model object."
        )
    if (
        config.modality != "text"
        and compatibility_config.mode != "native"
        and config.mode == "train"
        and config.fine_tune_method in {"gkd", "gold"}
    ):
        raise ValueError(
            "Non-native vision patch embedding modes are not supported for "
            "trainer-owned GKD or GOLD teacher models."
        )

    dimensions = compatibility_config.dimensions
    if not isinstance(dimensions, (list, ListConfig)) or len(dimensions) == 0:
        raise ValueError("vision_patch_embedding.dimensions must be a non-empty list.")
    normalized_dimensions = [int(dimension) for dimension in dimensions]
    if len(set(normalized_dimensions)) != len(normalized_dimensions):
        raise ValueError(
            "vision_patch_embedding.dimensions must not contain duplicates."
        )
    if any(dimension not in {2, 3} for dimension in normalized_dimensions):
        raise ValueError("vision_patch_embedding.dimensions supports only 2 and 3.")

    _validate_auto_probe_config(
        auto_probe_config=compatibility_config.auto_probe,
    )


def prepare_vision_patch_embedding_compatibility(
    model: nn.Module,
    config: DictConfig,
    model_role: str,
) -> Dict[str, Any]:
    validate_vision_patch_embedding_config(config=config)
    if config.modality == "text":
        return _build_not_applicable_plan(model_role=model_role)

    compatibility_config = config.vision_patch_embedding
    candidates = _discover_candidates(
        model=model,
        dimensions=set(int(value) for value in compatibility_config.dimensions),
    )
    if compatibility_config.mode == "linear" and len(candidates) == 0:
        raise ValueError(
            "vision_patch_embedding.mode=linear found no full-patch convolution candidates."
        )

    warning_messages = _build_no_candidate_warnings(
        requested_mode=compatibility_config.mode,
        candidate_count=len(candidates),
        model_role=model_role,
    )

    runtime_fingerprint = _build_runtime_fingerprint()
    probe_config = OmegaConf.to_container(
        compatibility_config.auto_probe,
        resolve=True,
    )
    signatures = _collect_signatures(candidates=candidates)
    probe_cache = {}
    local_probe_results = {}
    local_decisions = {}
    for signature_key, signature in signatures.items():
        if compatibility_config.mode == "auto":
            if signature_key not in probe_cache:
                probe_cache[signature_key] = run_vision_patch_embedding_probe(
                    signature=signature,
                    probe_config=probe_config,
                )
            probe_result = probe_cache[signature_key]
            local_probe_results[signature_key] = probe_result
            local_decisions[signature_key] = probe_result["decision"]
        else:
            local_decisions[signature_key] = compatibility_config.mode

    return {
        "_candidates": candidates,
        "_dimensions": set(int(value) for value in compatibility_config.dimensions),
        "_model": model,
        "_probe_cache": probe_cache,
        "_probe_config": probe_config,
        "scope": model_role,
        "requested_mode": compatibility_config.mode,
        "runtime_fingerprint": runtime_fingerprint,
        "local_probe_results": local_probe_results,
        "local_decisions": local_decisions,
        "warnings": warning_messages,
    }


def apply_vision_patch_embedding_compatibility(
    compatibility_plan: Dict[str, Any],
) -> Dict[str, Any]:
    if compatibility_plan["requested_mode"] == "not_applicable":
        return _build_not_applicable_result(
            model_role=compatibility_plan["scope"],
        )

    local_summary = _build_local_summary(compatibility_plan=compatibility_plan)
    rank_evidence = _gather_rank_evidence(local_summary=local_summary)
    _validate_rank_candidate_consistency(rank_evidence=rank_evidence)
    _raise_for_probe_failures(rank_evidence=rank_evidence)
    global_decisions = _resolve_global_decisions(rank_evidence=rank_evidence)

    candidates = compatibility_plan["_candidates"]
    changed_modules: List[nn.Module] = []
    try:
        for candidate in candidates:
            module = candidate["module"]
            decision = global_decisions[candidate["signature_key"]]
            if decision == "linear":
                if _is_linear_strategy_applied(module=module):
                    continue
                _apply_linear_strategy(
                    module=module,
                    signature_key=candidate["signature_key"],
                )
                changed_modules.append(module)
            else:
                _restore_original_forward(module=module)
    except Exception:
        for module in changed_modules:
            _restore_original_forward(module=module)
        raise

    resolved_mode = _resolve_applied_mode(global_decisions=global_decisions)
    return {
        "scope": compatibility_plan["scope"],
        "requested_mode": compatibility_plan["requested_mode"],
        "resolved_mode": resolved_mode,
        "selection_reason": _build_selection_reason(
            requested_mode=compatibility_plan["requested_mode"],
            resolved_mode=resolved_mode,
            candidate_count=len(candidates),
        ),
        "runtime_fingerprint": compatibility_plan["runtime_fingerprint"],
        "global_decisions": global_decisions,
        "modules": _build_module_metadata(
            candidates=candidates,
            global_decisions=global_decisions,
        ),
        "rank_evidence": rank_evidence,
        "warnings": compatibility_plan["warnings"],
        "distributed_consistent": True,
    }


def apply_trainer_vision_patch_embedding_compatibility(
    trainer: Any,
    compatibility_plan: Dict[str, Any],
) -> Dict[str, Any]:
    related_plans = _prepare_trainer_related_plans(
        trainer=trainer,
        compatibility_plan=compatibility_plan,
    )
    all_plans = [compatibility_plan, *related_plans]
    try:
        compatibility_results = [
            apply_vision_patch_embedding_compatibility(
                compatibility_plan=model_plan,
            )
            for model_plan in all_plans
        ]
    except Exception:
        for model_plan in all_plans:
            for candidate in model_plan["_candidates"]:
                _restore_original_forward(module=candidate["module"])
        raise

    primary_result = compatibility_results[0]
    primary_result["related_models"] = compatibility_results[1:]
    return primary_result


def _prepare_trainer_related_plans(
    trainer: Any,
    compatibility_plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if compatibility_plan["requested_mode"] in {"native", "not_applicable"}:
        return []

    seen_model_ids = {id(compatibility_plan["_model"])}
    seen_candidate_ids = {
        id(candidate["module"]) for candidate in compatibility_plan["_candidates"]
    }
    related_plans = []
    for attribute_name, model_role in [
        ("ref_model", "trainer_reference_model"),
        ("teacher_model", "trainer_teacher_model"),
    ]:
        related_model = getattr(
            trainer,
            attribute_name,
            None,
        )
        if (
            not isinstance(related_model, nn.Module)
            or id(related_model) in seen_model_ids
        ):
            continue
        related_plan = _prepare_related_model_plan(
            model=related_model,
            model_role=model_role,
            compatibility_plan=compatibility_plan,
        )
        unique_candidates = [
            candidate
            for candidate in related_plan["_candidates"]
            if id(candidate["module"]) not in seen_candidate_ids
        ]
        seen_model_ids.add(id(related_model))
        if len(unique_candidates) == 0:
            continue
        related_plan["_candidates"] = unique_candidates
        unique_signature_keys = {
            candidate["signature_key"] for candidate in unique_candidates
        }
        related_plan["local_probe_results"] = {
            signature_key: probe_result
            for signature_key, probe_result in related_plan[
                "local_probe_results"
            ].items()
            if signature_key in unique_signature_keys
        }
        related_plan["local_decisions"] = {
            signature_key: decision
            for signature_key, decision in related_plan["local_decisions"].items()
            if signature_key in unique_signature_keys
        }
        related_candidate_ids = {
            id(candidate["module"]) for candidate in unique_candidates
        }
        seen_candidate_ids.update(related_candidate_ids)
        related_plans.append(related_plan)
    return related_plans


def _prepare_related_model_plan(
    model: nn.Module,
    model_role: str,
    compatibility_plan: Dict[str, Any],
) -> Dict[str, Any]:
    requested_mode = compatibility_plan["requested_mode"]
    candidates = _discover_candidates(
        model=model,
        dimensions=compatibility_plan["_dimensions"],
    )
    if requested_mode == "linear" and len(candidates) == 0:
        raise ValueError(
            f"vision_patch_embedding.mode=linear found no full-patch convolution "
            f"candidates in {model_role}."
        )

    warning_messages = _build_no_candidate_warnings(
        requested_mode=requested_mode,
        candidate_count=len(candidates),
        model_role=model_role,
    )
    signatures = _collect_signatures(candidates=candidates)
    local_probe_results = {}
    local_decisions = {}
    probe_cache = compatibility_plan["_probe_cache"]
    for signature_key, signature in signatures.items():
        if requested_mode == "auto":
            if signature_key not in probe_cache:
                probe_cache[signature_key] = run_vision_patch_embedding_probe(
                    signature=signature,
                    probe_config=compatibility_plan["_probe_config"],
                )
            probe_result = probe_cache[signature_key]
            local_probe_results[signature_key] = probe_result
            local_decisions[signature_key] = probe_result["decision"]
        else:
            local_decisions[signature_key] = requested_mode

    return {
        "_candidates": candidates,
        "_dimensions": compatibility_plan["_dimensions"],
        "_model": model,
        "_probe_cache": probe_cache,
        "_probe_config": compatibility_plan["_probe_config"],
        "scope": model_role,
        "requested_mode": requested_mode,
        "runtime_fingerprint": _build_runtime_fingerprint(),
        "local_probe_results": local_probe_results,
        "local_decisions": local_decisions,
        "warnings": warning_messages,
    }


def _build_no_candidate_warnings(
    requested_mode: str,
    candidate_count: int,
    model_role: str,
) -> List[str]:
    if requested_mode != "auto" or candidate_count > 0:
        return []
    warning_message = (
        f"Vision patch embedding auto mode found no compatible full-patch Conv2d "
        f"or Conv3d candidates in {model_role}; continuing with native model behavior."
    )
    if (
        int(
            os.environ.get(
                "RANK",
                0,
            )
        )
        == 0
    ):
        warnings.warn(
            warning_message,
            RuntimeWarning,
            stacklevel=3,
        )
    return [warning_message]


def _validate_auto_probe_config(
    auto_probe_config: DictConfig,
) -> None:
    required_keys = {
        "startup_timeout_seconds",
        "operation_timeout_seconds",
        "warmup_iterations",
        "measurement_iterations",
        "patch_counts",
        "slowdown_ratio",
        "minimum_slowdown_milliseconds",
        "fp32_equivalence_atol",
        "fp32_equivalence_rtol",
        "runtime_equivalence_atol",
        "runtime_equivalence_rtol",
    }
    missing_keys = required_keys - set(auto_probe_config.keys())
    if len(missing_keys) > 0:
        raise ValueError(
            "vision_patch_embedding.auto_probe is missing required keys: "
            f"{sorted(missing_keys)}"
        )
    positive_values = {
        "startup_timeout_seconds": auto_probe_config.startup_timeout_seconds,
        "operation_timeout_seconds": auto_probe_config.operation_timeout_seconds,
        "warmup_iterations": auto_probe_config.warmup_iterations,
        "measurement_iterations": auto_probe_config.measurement_iterations,
        "slowdown_ratio": auto_probe_config.slowdown_ratio,
        "fp32_equivalence_atol": auto_probe_config.fp32_equivalence_atol,
        "fp32_equivalence_rtol": auto_probe_config.fp32_equivalence_rtol,
        "runtime_equivalence_atol": auto_probe_config.runtime_equivalence_atol,
        "runtime_equivalence_rtol": auto_probe_config.runtime_equivalence_rtol,
    }
    for key, value in positive_values.items():
        if float(value) <= 0:
            raise ValueError(
                f"vision_patch_embedding.auto_probe.{key} must be positive."
            )
    if float(auto_probe_config.minimum_slowdown_milliseconds) < 0:
        raise ValueError(
            "vision_patch_embedding.auto_probe.minimum_slowdown_milliseconds "
            "must be greater than or equal to zero."
        )
    patch_counts = auto_probe_config.patch_counts
    if not isinstance(patch_counts, (list, ListConfig)) or len(patch_counts) == 0:
        raise ValueError(
            "vision_patch_embedding.auto_probe.patch_counts must be a non-empty list."
        )
    normalized_patch_counts = [int(value) for value in patch_counts]
    if any(value <= 0 for value in normalized_patch_counts):
        raise ValueError(
            "vision_patch_embedding.auto_probe.patch_counts must contain positive integers."
        )
    if normalized_patch_counts != sorted(set(normalized_patch_counts)):
        raise ValueError(
            "vision_patch_embedding.auto_probe.patch_counts must be unique and sorted."
        )


def _build_not_applicable_plan(
    model_role: str,
) -> Dict[str, Any]:
    return {
        "_candidates": [],
        "scope": model_role,
        "requested_mode": "not_applicable",
        "runtime_fingerprint": {},
        "local_probe_results": {},
        "local_decisions": {},
        "warnings": [],
    }


def _build_not_applicable_result(
    model_role: str,
) -> Dict[str, Any]:
    return {
        "scope": model_role,
        "requested_mode": "not_applicable",
        "resolved_mode": "not_applicable",
        "selection_reason": "text_model",
        "runtime_fingerprint": {},
        "global_decisions": {},
        "modules": [],
        "rank_evidence": [],
        "warnings": [],
        "distributed_consistent": True,
    }


def _discover_candidates(
    model: nn.Module,
    dimensions: Set[int],
) -> List[Dict[str, Any]]:
    candidates = []
    for module_path, module in model.named_modules():
        dimension = _get_convolution_dimension(module=module)
        if dimension is None or dimension not in dimensions:
            continue
        validation_error = _validate_candidate(
            module=module,
            dimension=dimension,
        )
        if validation_error is not None:
            continue
        signature = _build_signature(
            module=module,
            dimension=dimension,
        )
        candidates.append(
            {
                "path": module_path,
                "class": _get_module_class_path(module=module),
                "module": module,
                "signature": signature,
                "signature_key": _build_signature_key(signature=signature),
                "structure": _build_structure_fingerprint(
                    module=module,
                    dimension=dimension,
                ),
            }
        )
    return candidates


def _get_convolution_dimension(
    module: nn.Module,
) -> Optional[int]:
    if isinstance(module, nn.Conv2d) and module.__class__.forward is nn.Conv2d.forward:
        return 2
    if isinstance(module, nn.Conv3d) and module.__class__.forward is nn.Conv3d.forward:
        return 3
    return None


def _validate_candidate(
    module: nn.Module,
    dimension: int,
) -> Optional[str]:
    logical_shape = _resolve_parameter_shape(parameter=module.weight)
    if len(logical_shape) != dimension + 2:
        return "weight_rank_mismatch"
    if tuple(module.kernel_size) != tuple(module.stride):
        return "kernel_stride_mismatch"
    if not _is_zero_padding(
        padding=module.padding,
        dimension=dimension,
    ):
        return "nonzero_padding"
    if tuple(module.dilation) != (1,) * dimension:
        return "nonunit_dilation"
    if module.groups != 1:
        return "grouped_convolution"
    if logical_shape[0] != module.out_channels:
        return "output_channel_mismatch"
    if logical_shape[1] != module.in_channels:
        return "input_channel_mismatch"
    if logical_shape[2:] != tuple(module.kernel_size):
        return "kernel_shape_mismatch"
    if module.bias is not None and _resolve_parameter_shape(
        parameter=module.bias,
    ) != (module.out_channels,):
        return "bias_shape_mismatch"
    return None


def _is_zero_padding(
    padding: Any,
    dimension: int,
) -> bool:
    if padding == "valid":
        return True
    if isinstance(padding, str):
        return False
    return tuple(padding) == (0,) * dimension


def _build_signature(
    module: nn.Module,
    dimension: int,
) -> Dict[str, Any]:
    return {
        "dimension": dimension,
        "in_channels": module.in_channels,
        "out_channels": module.out_channels,
        "kernel_size": list(module.kernel_size),
        "stride": list(module.stride),
        "bias": module.bias is not None,
        "dtype": str(module.weight.dtype),
    }


def _build_signature_key(
    signature: Dict[str, Any],
) -> str:
    return json.dumps(
        signature,
        sort_keys=True,
        separators=(",", ":"),
    )


def _collect_signatures(
    candidates: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    return {
        candidate["signature_key"]: candidate["signature"] for candidate in candidates
    }


def _resolve_parameter_shape(
    parameter: torch.Tensor,
) -> Tuple[int, ...]:
    if hasattr(
        parameter,
        "ds_shape",
    ):
        return tuple(parameter.ds_shape)
    return tuple(parameter.shape)


def _build_structure_fingerprint(
    module: nn.Module,
    dimension: int,
) -> Dict[str, Any]:
    return {
        "dimension": dimension,
        "in_channels": module.in_channels,
        "out_channels": module.out_channels,
        "kernel_size": list(module.kernel_size),
        "stride": list(module.stride),
        "padding": (
            module.padding
            if isinstance(
                module.padding,
                str,
            )
            else list(module.padding)
        ),
        "dilation": list(module.dilation),
        "groups": module.groups,
        "bias": module.bias is not None,
        "weight_shape": list(_resolve_parameter_shape(parameter=module.weight)),
    }


def _get_module_class_path(
    module: nn.Module,
) -> str:
    return f"{module.__class__.__module__}.{module.__class__.__qualname__}"


def _build_runtime_fingerprint() -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            0,
        )
    )
    current_device = None
    if cuda_available:
        current_device = (
            local_rank
            if local_rank < torch.cuda.device_count()
            else torch.cuda.current_device()
        )
    capability = (
        torch.cuda.get_device_capability(current_device)
        if current_device is not None
        else None
    )
    cudnn_version = torch.backends.cudnn.version()
    return {
        "torch_version": torch.__version__,
        "transformers_version": _get_package_version(package_name="transformers"),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            _normalize_cudnn_version(cudnn_version=cudnn_version)
            if cudnn_version is not None
            else None
        ),
        "driver_version": _get_driver_version(),
        "device_name": (
            torch.cuda.get_device_name(current_device)
            if current_device is not None
            else None
        ),
        "compute_capability": (
            f"{capability[0]}.{capability[1]}" if capability is not None else None
        ),
    }


def _get_package_version(
    package_name: str,
) -> Optional[str]:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _normalize_cudnn_version(
    cudnn_version: int,
) -> str:
    major = cudnn_version // 10000
    minor = (cudnn_version % 10000) // 100
    patch = cudnn_version % 100
    return f"{major}.{minor}.{patch}"


def _get_driver_version() -> Optional[str]:
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        try:
            driver_version = pynvml.nvmlSystemGetDriverVersion()
        finally:
            pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        return None
    if isinstance(driver_version, bytes):
        return driver_version.decode("utf-8")
    return str(driver_version)


def _build_local_summary(
    compatibility_plan: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "rank": int(
            os.environ.get(
                "RANK",
                0,
            )
        ),
        "scope": compatibility_plan["scope"],
        "requested_mode": compatibility_plan["requested_mode"],
        "runtime_fingerprint": compatibility_plan["runtime_fingerprint"],
        "candidate_signatures": sorted(compatibility_plan["local_decisions"].keys()),
        "local_decisions": compatibility_plan["local_decisions"],
        "local_probe_results": compatibility_plan["local_probe_results"],
    }


def _gather_rank_evidence(
    local_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    environment_world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            1,
        )
    )
    if environment_world_size > 1 and not dist.is_initialized():
        raise RuntimeError(
            "Vision patch embedding distributed resolution requires an initialized "
            "torch process group."
        )
    if not dist.is_available() or not dist.is_initialized():
        return [local_summary]

    gathered_summaries: List[Optional[Dict[str, Any]]] = [None] * dist.get_world_size()
    dist.all_gather_object(
        gathered_summaries,
        local_summary,
    )
    return [summary for summary in gathered_summaries if summary is not None]


def _validate_rank_candidate_consistency(
    rank_evidence: List[Dict[str, Any]],
) -> None:
    reference = rank_evidence[0]
    mismatched_ranks = [
        evidence["rank"]
        for evidence in rank_evidence[1:]
        if evidence["scope"] != reference["scope"]
        or evidence["requested_mode"] != reference["requested_mode"]
        or evidence["candidate_signatures"] != reference["candidate_signatures"]
    ]
    if len(mismatched_ranks) > 0:
        raise RuntimeError(
            "Vision patch embedding candidates differ across distributed ranks: "
            f"mismatched_ranks={mismatched_ranks}."
        )


def _raise_for_probe_failures(
    rank_evidence: List[Dict[str, Any]],
) -> None:
    failures = [
        {
            "rank": evidence["rank"],
            "signature": signature,
            "result": result,
        }
        for evidence in rank_evidence
        for signature, result in evidence["local_probe_results"].items()
        if result["decision"] == "error"
    ]
    if len(failures) > 0:
        raise RuntimeError(
            "Vision patch embedding auto probe failed without valid native or "
            f"linear evidence: {failures}"
        )


def _resolve_global_decisions(
    rank_evidence: List[Dict[str, Any]],
) -> Dict[str, str]:
    signatures = rank_evidence[0]["candidate_signatures"]
    return {
        signature: (
            "linear"
            if any(
                evidence["local_decisions"][signature] == "linear"
                for evidence in rank_evidence
            )
            else "native"
        )
        for signature in signatures
    }


def _apply_linear_strategy(
    module: nn.Module,
    signature_key: str,
) -> None:
    setattr(
        module,
        "_vision_patch_embedding_original_forward",
        module.forward,
    )
    setattr(
        module,
        "_vision_patch_embedding_signature_key",
        signature_key,
    )
    module.forward = MethodType(
        _full_patch_convolution_forward,
        module,
    )


def _restore_original_forward(
    module: nn.Module,
) -> None:
    if not hasattr(
        module,
        "_vision_patch_embedding_original_forward",
    ):
        return
    module.forward = module._vision_patch_embedding_original_forward
    delattr(
        module,
        "_vision_patch_embedding_original_forward",
    )
    if hasattr(
        module,
        "_vision_patch_embedding_signature_key",
    ):
        delattr(
            module,
            "_vision_patch_embedding_signature_key",
        )


def _is_linear_strategy_applied(
    module: nn.Module,
) -> bool:
    return (
        getattr(
            module.forward,
            "__func__",
            None,
        )
        is _full_patch_convolution_forward
    )


def _full_patch_convolution_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(hidden_states, torch.Tensor):
        return module._vision_patch_embedding_original_forward(hidden_states)
    dimension = _get_convolution_dimension(module=module)
    if dimension is None or hidden_states.ndim != dimension + 2:
        return module._vision_patch_embedding_original_forward(hidden_states)
    if hidden_states.shape[1] != module.in_channels:
        return module._vision_patch_embedding_original_forward(hidden_states)
    if tuple(hidden_states.shape[2:]) != tuple(module.kernel_size):
        return module._vision_patch_embedding_original_forward(hidden_states)
    if module.weight.ndim != dimension + 2:
        raise RuntimeError(
            "Vision patch embedding convolution weight was not materialized before "
            "the linear forward."
        )

    projected_states = F.linear(
        hidden_states.flatten(1),
        module.weight.flatten(1),
        module.bias,
    )
    return projected_states.view(
        projected_states.shape[0],
        module.out_channels,
        *((1,) * dimension),
    )


def _resolve_applied_mode(
    global_decisions: Dict[str, str],
) -> str:
    decisions = set(global_decisions.values())
    if len(decisions) == 0 or decisions == {"native"}:
        return "native"
    if decisions == {"linear"}:
        return "linear"
    return "mixed"


def _build_selection_reason(
    requested_mode: str,
    resolved_mode: str,
    candidate_count: int,
) -> str:
    if candidate_count == 0:
        return "no_full_patch_candidates"
    if requested_mode == "native":
        return "explicit_native"
    if requested_mode == "linear":
        return "explicit_linear"
    if resolved_mode == "native":
        return "auto_native"
    if resolved_mode == "linear":
        return "auto_linear"
    return "auto_mixed"


def _build_module_metadata(
    candidates: List[Dict[str, Any]],
    global_decisions: Dict[str, str],
) -> List[Dict[str, Any]]:
    return [
        {
            "path": candidate["path"],
            "class": candidate["class"],
            "signature": candidate["signature"],
            "structure": candidate["structure"],
            "decision": global_decisions[candidate["signature_key"]],
            "applied": global_decisions[candidate["signature_key"]] == "linear",
        }
        for candidate in candidates
    ]
