from typing import Dict, List, Tuple, Set, Optional, Any
import importlib.metadata
from types import MethodType
import warnings

from omegaconf import DictConfig, ListConfig

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

import torch
from torch import distributed as dist
from torch import nn
import torch.nn.functional as F


def validate_vision_patch_embedding_config(
    config: DictConfig,
) -> None:
    compatibility_config = config.vision_patch_embedding
    if compatibility_config.mode not in {"conv3d", "linear", "auto"}:
        raise ValueError("vision_patch_embedding.mode must be conv3d, linear, or auto.")
    if compatibility_config.mode == "linear" and config.modality == "text":
        raise ValueError(
            "vision_patch_embedding.mode=linear requires an image-text model."
        )
    if (
        config.modality != "text"
        and compatibility_config.mode != "conv3d"
        and config.mode in {"test_vllm", "test_vllm_multi_turn"}
    ):
        raise ValueError(
            "Non-conv3d vision patch embedding modes are not supported by vLLM test modes."
        )
    if (
        config.modality != "text"
        and compatibility_config.mode != "conv3d"
        and config.fine_tune_method == "async_grpo"
    ):
        raise ValueError(
            "Non-conv3d vision patch embedding modes are not supported by async GRPO."
        )
    if (
        config.modality != "text"
        and compatibility_config.mode != "conv3d"
        and config.fine_tune_method in {"gkd", "gold"}
    ):
        raise ValueError(
            "Non-conv3d vision patch embedding modes are not supported for trainer-owned teacher models."
        )

    target_rules = compatibility_config.target_rules
    runtime_rules = compatibility_config.runtime_rules
    if not isinstance(target_rules, (list, ListConfig)):
        raise ValueError("vision_patch_embedding.target_rules must be a list.")
    if not isinstance(runtime_rules, (list, ListConfig)):
        raise ValueError("vision_patch_embedding.runtime_rules must be a list.")
    if compatibility_config.mode == "linear" and len(target_rules) == 0:
        raise ValueError(
            "vision_patch_embedding.mode=linear requires at least one target rule."
        )
    if compatibility_config.mode == "auto" and (
        len(target_rules) == 0 or len(runtime_rules) == 0
    ):
        raise ValueError(
            "vision_patch_embedding.mode=auto requires target and runtime rules."
        )

    _validate_target_rules(target_rules=target_rules)
    _validate_runtime_rules(runtime_rules=runtime_rules)


def apply_vision_patch_embedding_compatibility(
    model: nn.Module,
    config: DictConfig,
    rank: int,
    model_role: str,
) -> Dict[str, Any]:
    validate_vision_patch_embedding_config(config=config)
    compatibility_config = config.vision_patch_embedding
    if config.modality == "text":
        return _build_not_applicable_result(model_role=model_role)

    runtime_fingerprint = _build_runtime_fingerprint()
    warnings_payload: List[str] = []
    if (
        runtime_fingerprint["device_name"] is not None
        and runtime_fingerprint["driver_version"] is None
    ):
        warnings_payload.append(
            "Vision patch embedding could not observe the NVIDIA driver version; "
            "runtime rules that require a driver version will not match."
        )
    matched_runtime_rules = _find_matching_runtime_rules(
        runtime_rules=compatibility_config.runtime_rules,
        runtime_fingerprint=runtime_fingerprint,
    )
    target_inspections = _inspect_model_targets(
        model=model,
        target_rules=compatibility_config.target_rules,
    )
    unverified_conv3d_modules = _find_unverified_conv3d_modules(
        model=model,
        target_inspections=target_inspections,
    )
    resolved_mode, selection_reason = _resolve_mode(
        requested_mode=compatibility_config.mode,
        matched_runtime_rules=matched_runtime_rules,
        target_inspections=target_inspections,
        unverified_conv3d_modules=unverified_conv3d_modules,
        warnings_payload=warnings_payload,
    )

    if resolved_mode == "linear":
        _apply_linear_strategy(target_inspections=target_inspections)
    else:
        _restore_original_forwards(target_inspections=target_inspections)

    for message in warnings_payload:
        if rank == 0:
            warnings.warn(
                message,
                RuntimeWarning,
                stacklevel=2,
            )

    return {
        "scope": model_role,
        "resolved_mode": resolved_mode,
        "selection_reason": selection_reason,
        "matched_runtime_rules": matched_runtime_rules,
        "runtime_fingerprint": runtime_fingerprint,
        "modules": _build_module_metadata(
            target_inspections=target_inspections,
            resolved_mode=resolved_mode,
        ),
        "unverified_conv3d_modules": unverified_conv3d_modules,
        "warnings": warnings_payload,
        "distributed_consistent": None,
    }


def validate_distributed_vision_patch_embedding_result(
    compatibility_result: Dict[str, Any],
) -> Dict[str, Any]:
    validated_result = dict(compatibility_result)
    if not dist.is_available() or not dist.is_initialized():
        validated_result["distributed_consistent"] = True
        return validated_result

    local_summary = _build_distributed_summary(
        compatibility_result=compatibility_result,
    )
    gathered_summaries: List[Optional[Dict[str, Any]]] = [None] * dist.get_world_size()
    dist.all_gather_object(
        gathered_summaries,
        local_summary,
    )
    reference_summary = gathered_summaries[0]
    if any(summary != reference_summary for summary in gathered_summaries[1:]):
        raise RuntimeError(
            "Vision patch embedding compatibility differs across distributed ranks: "
            f"{gathered_summaries}"
        )

    validated_result["distributed_consistent"] = True
    return validated_result


def _validate_target_rules(
    target_rules: ListConfig,
) -> None:
    rule_names: Set[str] = set()
    class_version_pairs: Set[Tuple[str, str]] = set()
    installed_class_rules: Dict[str, List[str]] = {}
    required_keys = {
        "name",
        "strategy",
        "projection_path",
        "package_name",
        "package_version",
        "module_classes",
    }
    for rule_index, rule in enumerate(target_rules):
        if not isinstance(rule, DictConfig):
            raise ValueError(
                f"vision_patch_embedding.target_rules[{rule_index}] must be a mapping."
            )
        _require_config_keys(
            config_value=rule,
            required_keys=required_keys,
            config_path=f"vision_patch_embedding.target_rules[{rule_index}]",
        )
        _validate_trimmed_string(
            value=rule.name,
            config_path=f"vision_patch_embedding.target_rules[{rule_index}].name",
        )
        if rule.name in rule_names:
            raise ValueError(f"Duplicate vision patch target rule name: {rule.name}")
        rule_names.add(rule.name)
        if rule.strategy != "flattened_full_patch":
            raise ValueError(
                f"Unsupported vision patch embedding strategy: {rule.strategy}"
            )
        _validate_trimmed_string(
            value=rule.projection_path,
            config_path=(
                f"vision_patch_embedding.target_rules[{rule_index}].projection_path"
            ),
        )
        if not rule.projection_path.isidentifier():
            raise ValueError(
                "vision_patch_embedding projection_path must be one attribute name."
            )
        _validate_trimmed_string(
            value=rule.package_name,
            config_path=(
                f"vision_patch_embedding.target_rules[{rule_index}].package_name"
            ),
        )
        _validate_version_specifier(
            value=rule.package_version,
            config_path=(
                f"vision_patch_embedding.target_rules[{rule_index}].package_version"
            ),
        )
        installed_package_version = _get_package_version(
            package_name=rule.package_name,
        )
        if (
            not isinstance(rule.module_classes, (list, ListConfig))
            or len(rule.module_classes) == 0
        ):
            raise ValueError(
                "vision_patch_embedding target rule module_classes must be a non-empty list."
            )
        for module_class in rule.module_classes:
            _validate_trimmed_string(
                value=module_class,
                config_path=(
                    f"vision_patch_embedding.target_rules[{rule_index}].module_classes"
                ),
            )
            class_version_pair = (
                module_class,
                rule.package_version,
            )
            if class_version_pair in class_version_pairs:
                raise ValueError(
                    "Duplicate vision patch target class and package version: "
                    f"{module_class} {rule.package_version}"
                )
            class_version_pairs.add(class_version_pair)
            if installed_package_version is not None and _version_matches(
                version_specifier=rule.package_version,
                observed_version=installed_package_version,
            ):
                installed_class_rules.setdefault(
                    module_class,
                    [],
                ).append(rule.name)

    for module_class, matching_rule_names in installed_class_rules.items():
        if len(matching_rule_names) > 1:
            raise ValueError(
                "Multiple vision patch target rules match the installed package "
                f"version for {module_class}: {sorted(matching_rule_names)}"
            )


def _validate_runtime_rules(
    runtime_rules: ListConfig,
) -> None:
    rule_names: Set[str] = set()
    required_keys = {
        "name",
        "torch_version",
        "cuda_version",
        "cudnn_version",
        "driver_version",
        "compute_capabilities",
    }
    for rule_index, rule in enumerate(runtime_rules):
        if not isinstance(rule, DictConfig):
            raise ValueError(
                f"vision_patch_embedding.runtime_rules[{rule_index}] must be a mapping."
            )
        _require_config_keys(
            config_value=rule,
            required_keys=required_keys,
            config_path=f"vision_patch_embedding.runtime_rules[{rule_index}]",
        )
        _validate_trimmed_string(
            value=rule.name,
            config_path=f"vision_patch_embedding.runtime_rules[{rule_index}].name",
        )
        if rule.name in rule_names:
            raise ValueError(f"Duplicate vision patch runtime rule name: {rule.name}")
        rule_names.add(rule.name)
        for version_key in [
            "torch_version",
            "cuda_version",
            "cudnn_version",
            "driver_version",
        ]:
            _validate_version_specifier(
                value=rule[version_key],
                config_path=(
                    f"vision_patch_embedding.runtime_rules[{rule_index}].{version_key}"
                ),
            )
        if (
            not isinstance(rule.compute_capabilities, (list, ListConfig))
            or len(rule.compute_capabilities) == 0
        ):
            raise ValueError(
                "vision_patch_embedding runtime rule compute_capabilities must be a non-empty list."
            )
        for capability in rule.compute_capabilities:
            _validate_trimmed_string(
                value=capability,
                config_path=(
                    "vision_patch_embedding.runtime_rules"
                    f"[{rule_index}].compute_capabilities"
                ),
            )


def _require_config_keys(
    config_value: DictConfig,
    required_keys: Set[str],
    config_path: str,
) -> None:
    missing_keys = required_keys - set(config_value.keys())
    if len(missing_keys) > 0:
        raise ValueError(
            f"{config_path} is missing required keys: {sorted(missing_keys)}"
        )


def _validate_trimmed_string(
    value: str,
    config_path: str,
) -> None:
    if not isinstance(value, str) or value.strip() == "" or value != value.strip():
        raise ValueError(f"{config_path} must be a trimmed non-empty string.")


def _validate_version_specifier(
    value: str,
    config_path: str,
) -> None:
    _validate_trimmed_string(
        value=value,
        config_path=config_path,
    )
    try:
        SpecifierSet(value)
    except InvalidSpecifier as error:
        raise ValueError(
            f"Invalid version specifier at {config_path}: {value}"
        ) from error


def _build_not_applicable_result(
    model_role: str,
) -> Dict[str, Any]:
    return {
        "scope": model_role,
        "resolved_mode": "conv3d",
        "selection_reason": "not_applicable_text_model",
        "matched_runtime_rules": [],
        "runtime_fingerprint": {},
        "modules": [],
        "unverified_conv3d_modules": [],
        "warnings": [],
        "distributed_consistent": True,
    }


def _build_runtime_fingerprint() -> Dict[str, Optional[str]]:
    cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
    current_device = torch.cuda.current_device() if cuda_available else None
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


def _find_matching_runtime_rules(
    runtime_rules: ListConfig,
    runtime_fingerprint: Dict[str, Optional[str]],
) -> List[str]:
    return sorted(
        rule.name
        for rule in runtime_rules
        if _matches_runtime_rule(
            rule=rule,
            runtime_fingerprint=runtime_fingerprint,
        )
    )


def _matches_runtime_rule(
    rule: DictConfig,
    runtime_fingerprint: Dict[str, Optional[str]],
) -> bool:
    version_pairs = [
        (rule.torch_version, runtime_fingerprint["torch_version"]),
        (rule.cuda_version, runtime_fingerprint["cuda_version"]),
        (rule.cudnn_version, runtime_fingerprint["cudnn_version"]),
        (rule.driver_version, runtime_fingerprint["driver_version"]),
    ]
    for version_specifier, observed_version in version_pairs:
        if observed_version is None or not _version_matches(
            version_specifier=version_specifier,
            observed_version=observed_version,
        ):
            return False
    return runtime_fingerprint["compute_capability"] in set(rule.compute_capabilities)


def _version_matches(
    version_specifier: str,
    observed_version: str,
) -> bool:
    try:
        return SpecifierSet(version_specifier).contains(
            Version(observed_version),
            prereleases=True,
        )
    except InvalidVersion:
        return False


def _inspect_model_targets(
    model: nn.Module,
    target_rules: ListConfig,
) -> List[Dict[str, Any]]:
    inspections: List[Dict[str, Any]] = []
    for module_path, module in model.named_modules():
        module_class = _get_module_class_path(module=module)
        matching_rules = _find_matching_target_rules(
            module_class=module_class,
            target_rules=target_rules,
        )
        if len(matching_rules) == 0:
            continue
        if len(matching_rules) > 1:
            raise ValueError(
                "Multiple vision patch target rules match the current module: "
                f"module={module_path}, class={module_class}, "
                f"rules={[rule.name for rule in matching_rules]}"
            )
        inspections.append(
            _inspect_target_module(
                module_path=module_path,
                module=module,
                target_rule=matching_rules[0],
            )
        )
    return inspections


def _get_module_class_path(
    module: nn.Module,
) -> str:
    return f"{module.__class__.__module__}.{module.__class__.__qualname__}"


def _find_matching_target_rules(
    module_class: str,
    target_rules: ListConfig,
) -> List[DictConfig]:
    matching_rules = []
    for rule in target_rules:
        if module_class not in set(rule.module_classes):
            continue
        package_version = _get_package_version(package_name=rule.package_name)
        if package_version is None or not _version_matches(
            version_specifier=rule.package_version,
            observed_version=package_version,
        ):
            continue
        matching_rules.append(rule)
    return matching_rules


def _inspect_target_module(
    module_path: str,
    module: nn.Module,
    target_rule: DictConfig,
) -> Dict[str, Any]:
    projection = (
        getattr(
            module,
            target_rule.projection_path,
        )
        if hasattr(
            module,
            target_rule.projection_path,
        )
        else None
    )
    validation_error = _validate_projection(projection=projection)
    return {
        "path": module_path,
        "module": module,
        "class": _get_module_class_path(module=module),
        "target_rule": target_rule.name,
        "strategy": target_rule.strategy,
        "projection_path": target_rule.projection_path,
        "projection": projection,
        "structure": _build_structure_fingerprint(projection=projection),
        "validation_error": validation_error,
    }


def _validate_projection(
    projection: Any,
) -> Optional[str]:
    if not isinstance(projection, nn.Conv3d):
        return "projection is not torch.nn.Conv3d"
    weight_shape = _resolve_parameter_shape(parameter=projection.weight)
    if len(weight_shape) != 5:
        return "projection weight must have rank 5"
    if projection.kernel_size != projection.stride:
        return "projection kernel_size and stride must match"
    if projection.padding != (0, 0, 0):
        return "projection padding must be zero"
    if projection.dilation != (1, 1, 1):
        return "projection dilation must be one"
    if projection.groups != 1:
        return "projection groups must be one"
    if weight_shape[0] != projection.out_channels:
        return "projection weight output channels do not match"
    if weight_shape[1] != projection.in_channels:
        return "projection weight input channels do not match"
    if weight_shape[2:] != projection.kernel_size:
        return "projection weight kernel shape does not match"
    if projection.bias is not None and _resolve_parameter_shape(
        parameter=projection.bias,
    ) != (projection.out_channels,):
        return "projection bias shape does not match output channels"
    return None


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
    projection: Any,
) -> Dict[str, Any]:
    if not isinstance(projection, nn.Conv3d):
        return {
            "projection_class": (
                _get_module_class_path(module=projection)
                if isinstance(projection, nn.Module)
                else None
            ),
        }
    return {
        "projection_class": _get_module_class_path(module=projection),
        "in_channels": projection.in_channels,
        "out_channels": projection.out_channels,
        "kernel_size": list(projection.kernel_size),
        "stride": list(projection.stride),
        "padding": list(projection.padding),
        "dilation": list(projection.dilation),
        "groups": projection.groups,
        "bias": projection.bias is not None,
        "weight_shape": list(
            _resolve_parameter_shape(parameter=projection.weight),
        ),
    }


def _find_unverified_conv3d_modules(
    model: nn.Module,
    target_inspections: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    verified_projection_paths = {
        _join_module_path(
            module_path=inspection["path"],
            child_path=inspection["projection_path"],
        )
        for inspection in target_inspections
    }
    return [
        {
            "path": module_path,
            "class": _get_module_class_path(module=module),
        }
        for module_path, module in model.named_modules()
        if isinstance(module, nn.Conv3d)
        and module_path not in verified_projection_paths
    ]


def _join_module_path(
    module_path: str,
    child_path: str,
) -> str:
    if module_path == "":
        return child_path
    return f"{module_path}.{child_path}"


def _resolve_mode(
    requested_mode: str,
    matched_runtime_rules: List[str],
    target_inspections: List[Dict[str, Any]],
    unverified_conv3d_modules: List[Dict[str, str]],
    warnings_payload: List[str],
) -> Tuple[str, str]:
    validation_errors = [
        (
            inspection["path"],
            inspection["class"],
            inspection["validation_error"],
        )
        for inspection in target_inspections
        if inspection["validation_error"] is not None
    ]
    if requested_mode == "linear":
        if len(target_inspections) == 0:
            raise ValueError(
                "vision_patch_embedding.mode=linear found no certified target modules."
            )
        if len(validation_errors) > 0:
            raise ValueError(
                "vision_patch_embedding.mode=linear found invalid target structures: "
                f"{validation_errors}"
            )
        return "linear", "explicit_linear"

    if len(matched_runtime_rules) == 0:
        return "conv3d", "runtime_rule_not_matched"

    if len(target_inspections) == 0:
        warnings_payload.append(
            "Vision patch embedding matched known-risk runtime rules "
            f"{matched_runtime_rules}, but no certified target module was found; "
            "continuing with the original Conv3d implementation. Unverified Conv3d "
            f"modules: {unverified_conv3d_modules}"
        )
        return "conv3d", "certified_target_not_found"

    if len(validation_errors) > 0:
        warnings_payload.append(
            "Vision patch embedding matched known-risk runtime rules "
            f"{matched_runtime_rules}, but certified target validation failed "
            f"with {validation_errors}; continuing with the original Conv3d implementation."
        )
        return "conv3d", "target_structure_validation_failed"

    if requested_mode == "conv3d":
        warnings_payload.append(
            "Vision patch embedding matched known-risk runtime rules "
            f"{matched_runtime_rules}, but mode=conv3d keeps the original Conv3d implementation."
        )
        return "conv3d", "explicit_conv3d"

    return "linear", "matched_runtime_and_target_rules"


def _apply_linear_strategy(
    target_inspections: List[Dict[str, Any]],
) -> None:
    changed_modules: List[nn.Module] = []
    try:
        for inspection in target_inspections:
            module = inspection["module"]
            projection = inspection["projection"]
            if _is_linear_strategy_applied(
                module=module,
                projection=projection,
            ):
                continue
            setattr(
                module,
                "_vision_patch_embedding_original_forward",
                module.forward,
            )
            changed_modules.append(module)
            setattr(
                module,
                "_vision_patch_embedding_projection_path",
                inspection["projection_path"],
            )
            setattr(
                projection,
                "_vision_patch_embedding_original_forward",
                projection.forward,
            )
            changed_modules.append(projection)
            projection.forward = MethodType(
                _flattened_full_patch_projection_forward,
                projection,
            )
            module.forward = MethodType(
                _flattened_full_patch_forward,
                module,
            )
    except Exception:
        for module in changed_modules:
            _restore_original_forward(module=module)
        raise


def _restore_original_forwards(
    target_inspections: List[Dict[str, Any]],
) -> None:
    for inspection in target_inspections:
        _restore_original_forward(module=inspection["module"])
        _restore_original_forward(module=inspection["projection"])


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
        "_vision_patch_embedding_projection_path",
    ):
        delattr(
            module,
            "_vision_patch_embedding_projection_path",
        )


def _is_linear_strategy_applied(
    module: nn.Module,
    projection: nn.Conv3d,
) -> bool:
    return (
        getattr(
            module.forward,
            "__func__",
            None,
        )
        is _flattened_full_patch_forward
        and getattr(
            projection.forward,
            "__func__",
            None,
        )
        is _flattened_full_patch_projection_forward
    )


def _flattened_full_patch_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(hidden_states, torch.Tensor):
        raise ValueError("Vision patch embedding input must be a torch.Tensor.")
    if hidden_states.ndim != 2:
        raise ValueError("Vision patch embedding input must have rank 2.")
    projection = getattr(
        module,
        module._vision_patch_embedding_projection_path,
    )
    expected_features = (
        projection.in_channels
        * projection.kernel_size[0]
        * projection.kernel_size[1]
        * projection.kernel_size[2]
    )
    if hidden_states.shape[-1] != expected_features:
        raise ValueError(
            "Vision patch embedding input feature dimension does not match the "
            f"full patch volume: expected={expected_features}, "
            f"observed={hidden_states.shape[-1]}."
        )
    return module._vision_patch_embedding_original_forward(hidden_states)


def _flattened_full_patch_projection_forward(
    projection: nn.Conv3d,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(hidden_states, torch.Tensor):
        raise ValueError("Vision patch embedding input must be a torch.Tensor.")
    if hidden_states.ndim != 5:
        raise ValueError("Vision patch embedding projection input must have rank 5.")
    expected_shape = (
        projection.in_channels,
        *projection.kernel_size,
    )
    if tuple(hidden_states.shape[1:]) != expected_shape:
        raise ValueError(
            "Vision patch embedding projection input shape does not match the "
            f"full patch volume: expected={expected_shape}, "
            f"observed={tuple(hidden_states.shape[1:])}."
        )
    if projection.weight.ndim != 5:
        raise RuntimeError(
            "Vision patch embedding projection weight was not materialized before "
            "the linear forward."
        )
    projected_states = F.linear(
        hidden_states.flatten(1).to(dtype=projection.weight.dtype),
        projection.weight.flatten(1),
        projection.bias,
    )
    return projected_states.view(
        -1,
        projection.out_channels,
        1,
        1,
        1,
    )


def _build_module_metadata(
    target_inspections: List[Dict[str, Any]],
    resolved_mode: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "path": inspection["path"],
            "class": inspection["class"],
            "target_rule": inspection["target_rule"],
            "strategy": inspection["strategy"],
            "projection_path": inspection["projection_path"],
            "structure": inspection["structure"],
            "validation_error": inspection["validation_error"],
            "applied": resolved_mode == "linear",
        }
        for inspection in target_inspections
    ]


def _build_distributed_summary(
    compatibility_result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "scope": compatibility_result["scope"],
        "resolved_mode": compatibility_result["resolved_mode"],
        "selection_reason": compatibility_result["selection_reason"],
        "matched_runtime_rules": compatibility_result["matched_runtime_rules"],
        "runtime_fingerprint": compatibility_result["runtime_fingerprint"],
        "modules": compatibility_result["modules"],
        "warnings": compatibility_result["warnings"],
    }
