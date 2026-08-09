from typing import Dict, List, Optional, Any
import os
from contextlib import nullcontext

from omegaconf import DictConfig, ListConfig, OmegaConf

from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from transformers import PreTrainedModel


def initialize_peft_model(
    model: PreTrainedModel,
    config: DictConfig,
    pretrained_model_name: str,
) -> PreTrainedModel:
    validate_peft_initialization_config(config=config)

    if not config.is_peft:
        return model

    mode = str(config.peft_initialization.mode)
    if mode == "fresh":
        return _initialize_fresh_peft_model(
            model=model,
            config=config,
        )
    if mode == "continue_from_adapter":
        return _continue_peft_model_from_adapter(
            model=model,
            config=config,
            pretrained_model_name=pretrained_model_name,
        )

    raise ValueError(f"Unsupported peft_initialization.mode: {mode}")


def build_peft_initialization_metadata(
    config: DictConfig,
) -> Dict[str, Any]:
    validate_peft_initialization_config(config=config)

    target_parameters = _normalize_target_parameters(
        value=OmegaConf.select(
            config,
            "peft_config.target_parameters",
        ),
    )
    uses_target_parameters = bool(config.is_peft) and len(target_parameters) > 0
    target_parameter_zero3_init_policy = _build_target_parameter_zero3_init_policy(
        config=config,
        target_parameters=target_parameters if uses_target_parameters else [],
    )

    metadata = {
        "mode": str(config.peft_initialization.mode),
        "is_peft": bool(config.is_peft),
        "uses_target_parameters": uses_target_parameters,
        "target_parameter_count": (
            len(target_parameters) if uses_target_parameters else 0
        ),
        "target_parameter_zero3_init_policy": target_parameter_zero3_init_policy,
        "requested_base_model_name": str(config.pretrained_model_name),
        "resolved_base_model_name_for_continuation": None,
        "adapter_path": None,
        "adapter_name": str(config.peft_initialization.adapter_name),
        "adapter_base_model_name_or_path": None,
        "current_peft_config_fingerprint": _build_current_peft_config_fingerprint(
            config=config,
        ),
        "adapter_config_fingerprint": None,
        "weighted_merge_base_reference": None,
        "weighted_merge_candidate": False,
    }

    if not is_peft_continue_from_adapter(config=config):
        return metadata

    adapter_path = _normalize_adapter_path(
        adapter_path=str(config.peft_initialization.adapter_path),
    )
    adapter_config = PeftConfig.from_pretrained(adapter_path)
    peft_config_dict = _build_peft_config_dict(config=config)

    _validate_adapter_base_model(
        adapter_config=adapter_config,
        pretrained_model_name=str(config.pretrained_model_name),
    )
    _validate_adapter_lora_config(
        adapter_config=adapter_config,
        peft_config_dict=peft_config_dict,
    )

    adapter_base_model = _get_adapter_base_model(adapter_config=adapter_config)
    metadata.update(
        {
            "resolved_base_model_name_for_continuation": str(
                config.pretrained_model_name
            ),
            "adapter_path": adapter_path,
            "adapter_base_model_name_or_path": adapter_base_model,
            "adapter_config_fingerprint": _build_adapter_config_fingerprint(
                adapter_config=adapter_config,
            ),
            "weighted_merge_base_reference": adapter_base_model,
            "weighted_merge_candidate": True,
        }
    )
    return metadata


def is_peft_continue_from_adapter(
    config: DictConfig,
) -> bool:
    return (
        bool(config.is_peft)
        and str(config.peft_initialization.mode) == "continue_from_adapter"
    )


def validate_peft_continuation_base_resolution(
    config: DictConfig,
) -> None:
    merged_model_path = _build_merged_model_auto_resolution_path(config=config)
    if merged_model_path is None:
        return
    if not os.path.exists(merged_model_path):
        return
    raise ValueError(
        "peft_initialization.mode=continue_from_adapter requires loading the "
        "original base model; merged_model_path auto-resolution is disabled. "
        "Set pretrained_model_name to the intended base model explicitly."
    )


def validate_peft_initialization_config(
    config: DictConfig,
) -> None:
    mode = str(config.peft_initialization.mode)
    if mode not in ["fresh", "continue_from_adapter"]:
        raise ValueError(
            "peft_initialization.mode must be fresh or continue_from_adapter."
        )

    if not config.is_peft and mode != "fresh":
        raise ValueError(
            "peft_initialization.mode=continue_from_adapter requires is_peft=true."
        )

    if mode == "continue_from_adapter":
        _validate_continue_from_adapter_config(config=config)


def has_peft_target_parameters(
    config: DictConfig,
) -> bool:
    if not config.is_peft:
        return False

    target_parameters = OmegaConf.select(
        config,
        "peft_config.target_parameters",
    )
    return len(_normalize_target_parameters(value=target_parameters)) > 0


def _build_target_parameter_zero3_init_policy(
    config: DictConfig,
    target_parameters: List[str],
) -> Optional[str]:
    if len(target_parameters) == 0:
        return None
    if config.strategy != "deepspeed":
        return None

    deepspeed_stage = OmegaConf.select(
        config,
        "deepspeed.zero_optimization.stage",
    )
    if deepspeed_stage is None or int(deepspeed_stage) != 3:
        return None

    zero3_init = str(config.model_loading.deepspeed.zero3_init)
    if zero3_init == "auto":
        return "hf_zero3_init_disabled_until_after_peft_target_parameter_wrapping"
    if zero3_init == "disabled":
        return "hf_zero3_init_disabled_by_config"
    return "unsupported_force_enabled"


def _validate_continue_from_adapter_config(
    config: DictConfig,
) -> None:
    if config.fine_tune_method == "async_grpo":
        raise ValueError(
            "peft_initialization.mode=continue_from_adapter is not supported "
            "with async_grpo in this release."
        )
    if config.peft_initialization.adapter_path is None:
        raise ValueError(
            "peft_initialization.adapter_path is required when "
            "peft_initialization.mode=continue_from_adapter."
        )
    if not config.peft_initialization.is_trainable:
        raise ValueError(
            "peft_initialization.is_trainable must be true for train-time "
            "adapter continuation."
        )
    if not config.peft_initialization.require_base_model_match:
        raise ValueError(
            "peft_initialization.require_base_model_match must remain true "
            "in this release."
        )
    if config.dense_to_moe.router_with_lora:
        raise ValueError(
            "dense_to_moe.router_with_lora is not supported with "
            "peft_initialization.mode=continue_from_adapter in this release."
        )
    validate_peft_continuation_base_resolution(config=config)


def _initialize_fresh_peft_model(
    model: PreTrainedModel,
    config: DictConfig,
) -> PreTrainedModel:
    peft_config_dict = _build_peft_config_dict(config=config)
    target_parameters = _normalize_target_parameters(
        value=peft_config_dict.get("target_parameters"),
    )
    with _build_peft_target_parameter_wrapping_context(
        model=model,
        target_parameters=target_parameters,
    ):
        _validate_peft_target_parameter_shapes(
            model=model,
            target_parameters=target_parameters,
        )
        peft_config = LoraConfig(**peft_config_dict)
        peft_model = get_peft_model(
            model=model,
            peft_config=peft_config,
        )
    _validate_trainable_lora_parameters(model=peft_model)
    return peft_model


def _continue_peft_model_from_adapter(
    model: PreTrainedModel,
    config: DictConfig,
    pretrained_model_name: str,
) -> PreTrainedModel:
    adapter_path = _normalize_adapter_path(
        adapter_path=str(config.peft_initialization.adapter_path),
    )
    adapter_config = PeftConfig.from_pretrained(adapter_path)
    peft_config_dict = _build_peft_config_dict(config=config)

    _validate_adapter_base_model(
        adapter_config=adapter_config,
        pretrained_model_name=pretrained_model_name,
    )
    _validate_adapter_lora_config(
        adapter_config=adapter_config,
        peft_config_dict=peft_config_dict,
    )
    target_parameters = _normalize_target_parameters(
        value=peft_config_dict.get("target_parameters"),
    )
    with _build_peft_target_parameter_wrapping_context(
        model=model,
        target_parameters=target_parameters,
    ):
        _validate_peft_target_parameter_shapes(
            model=model,
            target_parameters=target_parameters,
        )
        peft_model = PeftModel.from_pretrained(
            model=model,
            model_id=adapter_path,
            adapter_name=str(config.peft_initialization.adapter_name),
            is_trainable=bool(config.peft_initialization.is_trainable),
        )
    _validate_trainable_lora_parameters(model=peft_model)
    return peft_model


def _build_peft_config_dict(
    config: DictConfig,
) -> Dict[str, Any]:
    peft_config_dict = OmegaConf.to_container(
        config.peft_config,
        resolve=True,
    )
    if not isinstance(peft_config_dict, dict):
        raise TypeError("peft_config must resolve to a dictionary.")

    if config.dense_to_moe.router_with_lora:
        peft_config_dict["modules_to_save"] = ["gate"]

    return peft_config_dict


def _build_current_peft_config_fingerprint(
    config: DictConfig,
) -> Dict[str, Any]:
    if not config.is_peft:
        return {}
    return _build_peft_config_fingerprint(
        peft_config=_build_peft_config_dict(config=config),
    )


def _build_adapter_config_fingerprint(
    adapter_config: Any,
) -> Dict[str, Any]:
    return {
        "peft_type": _normalize_config_value(
            getattr(adapter_config, "peft_type", None)
        ),
        "task_type": _normalize_config_value(
            getattr(adapter_config, "task_type", None)
        ),
        "base_model_name_or_path": _normalize_config_value(
            getattr(adapter_config, "base_model_name_or_path", None)
        ),
        "r": _normalize_config_value(getattr(adapter_config, "r", None)),
        "lora_alpha": _normalize_config_value(
            getattr(adapter_config, "lora_alpha", None)
        ),
        "target_modules": _normalize_config_value(
            getattr(adapter_config, "target_modules", None)
        ),
        "target_parameters": _normalize_config_value(
            getattr(adapter_config, "target_parameters", None)
        ),
        "modules_to_save": _normalize_config_value(
            getattr(adapter_config, "modules_to_save", None)
        ),
        "bias": _normalize_config_value(getattr(adapter_config, "bias", None)),
        "inference_mode": _normalize_config_value(
            getattr(adapter_config, "inference_mode", None)
        ),
    }


def _build_peft_config_fingerprint(
    peft_config: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "peft_type": _normalize_config_value(peft_config.get("peft_type", "LORA")),
        "task_type": _normalize_config_value(peft_config.get("task_type")),
        "base_model_name_or_path": None,
        "r": _normalize_config_value(peft_config.get("r")),
        "lora_alpha": _normalize_config_value(peft_config.get("lora_alpha")),
        "target_modules": _normalize_config_value(peft_config.get("target_modules")),
        "target_parameters": _normalize_config_value(
            peft_config.get("target_parameters")
        ),
        "modules_to_save": _normalize_config_value(peft_config.get("modules_to_save")),
        "bias": _normalize_config_value(peft_config.get("bias")),
        "inference_mode": _normalize_config_value(peft_config.get("inference_mode")),
    }


def _normalize_adapter_path(
    adapter_path: str,
) -> str:
    normalized_adapter_path = os.path.normpath(adapter_path)
    if not os.path.isdir(normalized_adapter_path):
        raise ValueError(f"Adapter directory does not exist: {adapter_path}")

    adapter_config_path = os.path.join(
        normalized_adapter_path,
        "adapter_config.json",
    )
    if not os.path.isfile(adapter_config_path):
        raise ValueError(
            "adapter_config.json was not found. "
            f"adapter_path must point to a PEFT adapter directory: {adapter_path}"
        )

    return normalized_adapter_path


def _build_merged_model_auto_resolution_path(
    config: DictConfig,
) -> Optional[str]:
    if not config.is_preprocessed:
        return None

    pretrained_model_name = str(config.pretrained_model_name)
    if os.path.isabs(pretrained_model_name):
        return None

    return os.path.normpath(
        os.path.join(
            str(config.merged_model_path),
            pretrained_model_name,
        )
    )


def _get_adapter_base_model(
    adapter_config: Any,
) -> str:
    adapter_base_model = getattr(
        adapter_config,
        "base_model_name_or_path",
        None,
    )
    if not isinstance(adapter_base_model, str) or adapter_base_model == "":
        raise ValueError(
            "Adapter base_model_name_or_path is missing; cannot verify "
            "same-base continuation."
        )
    return adapter_base_model


def _validate_adapter_base_model(
    adapter_config: Any,
    pretrained_model_name: str,
) -> None:
    adapter_base_model = _get_adapter_base_model(adapter_config=adapter_config)

    if os.path.normpath(adapter_base_model) != os.path.normpath(pretrained_model_name):
        raise ValueError(
            "Adapter base model mismatch: "
            f"adapter={adapter_base_model}, current={pretrained_model_name}"
        )


def _validate_adapter_lora_config(
    adapter_config: Any,
    peft_config_dict: Dict[str, Any],
) -> None:
    _validate_equal_config(
        name="peft_type",
        adapter_value=getattr(adapter_config, "peft_type", None),
        expected_value=peft_config_dict.get("peft_type", "LORA"),
    )
    _validate_equal_config(
        name="task_type",
        adapter_value=getattr(adapter_config, "task_type", None),
        expected_value=peft_config_dict.get("task_type"),
    )
    _validate_equal_int_config(
        name="r",
        adapter_value=getattr(adapter_config, "r", None),
        expected_value=peft_config_dict.get("r"),
    )
    _validate_equal_int_config(
        name="lora_alpha",
        adapter_value=getattr(adapter_config, "lora_alpha", None),
        expected_value=peft_config_dict.get("lora_alpha"),
    )
    _validate_target_modules(
        adapter_value=getattr(adapter_config, "target_modules", None),
        expected_value=peft_config_dict.get("target_modules"),
    )
    _validate_target_parameters(
        adapter_value=getattr(adapter_config, "target_parameters", None),
        expected_value=peft_config_dict.get("target_parameters"),
    )
    _validate_equal_config(
        name="bias",
        adapter_value=getattr(adapter_config, "bias", None),
        expected_value=peft_config_dict.get("bias"),
    )
    _validate_equal_config(
        name="modules_to_save",
        adapter_value=getattr(adapter_config, "modules_to_save", None),
        expected_value=peft_config_dict.get("modules_to_save"),
    )


def _validate_equal_int_config(
    name: str,
    adapter_value: Any,
    expected_value: Any,
) -> None:
    if adapter_value is None or expected_value is None:
        raise ValueError(f"Adapter {name} compatibility cannot be verified.")
    if int(adapter_value) != int(expected_value):
        raise ValueError(
            f"Adapter {name} mismatch: adapter={adapter_value}, "
            f"expected={expected_value}"
        )


def _validate_equal_config(
    name: str,
    adapter_value: Any,
    expected_value: Any,
) -> None:
    normalized_adapter_value = _normalize_config_value(adapter_value)
    normalized_expected_value = _normalize_config_value(expected_value)
    if normalized_adapter_value != normalized_expected_value:
        raise ValueError(
            f"Adapter {name} mismatch: adapter={normalized_adapter_value}, "
            f"expected={normalized_expected_value}"
        )


def _validate_target_modules(
    adapter_value: Any,
    expected_value: Any,
) -> None:
    adapter_target_modules = _normalize_target_modules(value=adapter_value)
    expected_target_modules = _normalize_target_modules(value=expected_value)

    if expected_target_modules == ["all-linear"]:
        if len(adapter_target_modules) == 0:
            raise ValueError("Adapter target_modules is empty.")
        return

    if adapter_target_modules != expected_target_modules:
        raise ValueError(
            "Adapter target_modules mismatch: "
            f"adapter={adapter_target_modules}, expected={expected_target_modules}"
        )


def _normalize_target_modules(
    value: Any,
) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return sorted(str(item) for item in value)
    raise TypeError(f"Unsupported target_modules type: {type(value).__name__}")


def _validate_target_parameters(
    adapter_value: Any,
    expected_value: Any,
) -> None:
    adapter_target_parameters = _normalize_target_parameters(value=adapter_value)
    expected_target_parameters = _normalize_target_parameters(value=expected_value)

    if adapter_target_parameters != expected_target_parameters:
        raise ValueError(
            "Adapter target_parameters mismatch: "
            f"adapter={adapter_target_parameters}, expected={expected_target_parameters}"
        )


def _normalize_target_parameters(
    value: Any,
) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set, ListConfig)):
        return sorted(str(item) for item in value)
    raise TypeError(f"Unsupported target_parameters type: {type(value).__name__}")


def _build_peft_target_parameter_wrapping_context(
    model: PreTrainedModel,
    target_parameters: List[str],
) -> Any:
    if len(target_parameters) == 0:
        return nullcontext()

    named_target_parameters = _resolve_peft_target_parameter_objects(
        model=model,
        target_parameters=target_parameters,
    )
    if not _has_zero3_partitioned_target_parameters(
        named_target_parameters=named_target_parameters,
    ):
        return nullcontext()

    from deepspeed import zero

    return zero.GatheredParameters(
        list(named_target_parameters.values()),
        modifier_rank=None,
    )


def _resolve_peft_target_parameter_objects(
    model: PreTrainedModel,
    target_parameters: List[str],
) -> Dict[str, Any]:
    named_parameters = dict(model.named_parameters())
    missing_parameters = [
        target_parameter
        for target_parameter in target_parameters
        if target_parameter not in named_parameters
    ]
    if len(missing_parameters) > 0:
        raise ValueError(
            "PEFT target_parameters were not found in model.named_parameters(): "
            f"{missing_parameters[:10]}"
        )

    return {
        target_parameter: named_parameters[target_parameter]
        for target_parameter in target_parameters
    }


def _has_zero3_partitioned_target_parameters(
    named_target_parameters: Dict[str, Any],
) -> bool:
    return any(
        parameter.ndim == 1 and tuple(parameter.shape) == (0,)
        for parameter in named_target_parameters.values()
    )


def _validate_peft_target_parameter_shapes(
    model: PreTrainedModel,
    target_parameters: List[str],
) -> None:
    if len(target_parameters) == 0:
        return

    named_target_parameters = _resolve_peft_target_parameter_objects(
        model=model,
        target_parameters=target_parameters,
    )
    invalid_shapes = {
        target_parameter: tuple(parameter.shape)
        for target_parameter, parameter in named_target_parameters.items()
        if parameter.ndim not in [2, 3]
    }
    if len(invalid_shapes) > 0:
        raise ValueError(
            "PEFT target_parameters must resolve to full 2D or 3D tensors before "
            f"LoRA wrapping. Invalid shapes: {invalid_shapes}"
        )


def _normalize_config_value(
    value: Any,
) -> Any:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(
            value,
            resolve=True,
        )
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, (list, tuple, set, ListConfig)):
        return sorted(str(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _validate_trainable_lora_parameters(
    model: PreTrainedModel,
) -> None:
    trainable_lora_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.requires_grad
    )
    if trainable_lora_parameters == 0:
        raise RuntimeError(
            "No trainable LoRA parameters were found after loading "
            "the continuation adapter."
        )
