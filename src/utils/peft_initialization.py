from typing import Dict, List, Optional, Any
import os

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
        return initialize_fresh_peft_model(
            model=model,
            config=config,
        )
    if mode == "continue_from_adapter":
        return continue_peft_model_from_adapter(
            model=model,
            config=config,
            pretrained_model_name=pretrained_model_name,
        )

    raise ValueError(f"Unsupported peft_initialization.mode: {mode}")


def build_peft_initialization_metadata(
    config: DictConfig,
) -> Dict[str, Any]:
    validate_peft_initialization_config(config=config)

    metadata = {
        "mode": str(config.peft_initialization.mode),
        "is_peft": bool(config.is_peft),
        "requested_base_model_name": str(config.pretrained_model_name),
        "resolved_base_model_name_for_continuation": None,
        "adapter_path": None,
        "adapter_name": str(config.peft_initialization.adapter_name),
        "adapter_base_model_name_or_path": None,
        "current_peft_config_fingerprint": build_current_peft_config_fingerprint(
            config=config,
        ),
        "adapter_config_fingerprint": None,
        "weighted_merge_base_reference": None,
        "weighted_merge_candidate": False,
    }

    if not is_peft_continue_from_adapter(config=config):
        return metadata

    adapter_path = normalize_adapter_path(
        adapter_path=str(config.peft_initialization.adapter_path),
    )
    adapter_config = PeftConfig.from_pretrained(adapter_path)
    peft_config_dict = build_peft_config_dict(config=config)

    validate_adapter_base_model(
        adapter_config=adapter_config,
        pretrained_model_name=str(config.pretrained_model_name),
    )
    validate_adapter_lora_config(
        adapter_config=adapter_config,
        peft_config_dict=peft_config_dict,
    )

    adapter_base_model = get_adapter_base_model(adapter_config=adapter_config)
    metadata.update(
        {
            "resolved_base_model_name_for_continuation": str(
                config.pretrained_model_name
            ),
            "adapter_path": adapter_path,
            "adapter_base_model_name_or_path": adapter_base_model,
            "adapter_config_fingerprint": build_adapter_config_fingerprint(
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
    merged_model_path = build_merged_model_auto_resolution_path(config=config)
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
        validate_continue_from_adapter_config(config=config)


def validate_continue_from_adapter_config(
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


def initialize_fresh_peft_model(
    model: PreTrainedModel,
    config: DictConfig,
) -> PreTrainedModel:
    peft_config_dict = build_peft_config_dict(config=config)
    peft_config = LoraConfig(**peft_config_dict)
    return get_peft_model(
        model=model,
        peft_config=peft_config,
    )


def continue_peft_model_from_adapter(
    model: PreTrainedModel,
    config: DictConfig,
    pretrained_model_name: str,
) -> PreTrainedModel:
    adapter_path = normalize_adapter_path(
        adapter_path=str(config.peft_initialization.adapter_path),
    )
    adapter_config = PeftConfig.from_pretrained(adapter_path)
    peft_config_dict = build_peft_config_dict(config=config)

    validate_adapter_base_model(
        adapter_config=adapter_config,
        pretrained_model_name=pretrained_model_name,
    )
    validate_adapter_lora_config(
        adapter_config=adapter_config,
        peft_config_dict=peft_config_dict,
    )

    peft_model = PeftModel.from_pretrained(
        model=model,
        model_id=adapter_path,
        adapter_name=str(config.peft_initialization.adapter_name),
        is_trainable=bool(config.peft_initialization.is_trainable),
    )
    validate_trainable_lora_parameters(model=peft_model)
    return peft_model


def build_peft_config_dict(
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


def build_current_peft_config_fingerprint(
    config: DictConfig,
) -> Dict[str, Any]:
    if not config.is_peft:
        return {}
    return build_peft_config_fingerprint(
        peft_config=build_peft_config_dict(config=config),
    )


def build_adapter_config_fingerprint(
    adapter_config: Any,
) -> Dict[str, Any]:
    return {
        "peft_type": normalize_config_value(getattr(adapter_config, "peft_type", None)),
        "task_type": normalize_config_value(getattr(adapter_config, "task_type", None)),
        "base_model_name_or_path": normalize_config_value(
            getattr(adapter_config, "base_model_name_or_path", None)
        ),
        "r": normalize_config_value(getattr(adapter_config, "r", None)),
        "lora_alpha": normalize_config_value(
            getattr(adapter_config, "lora_alpha", None)
        ),
        "target_modules": normalize_config_value(
            getattr(adapter_config, "target_modules", None)
        ),
        "modules_to_save": normalize_config_value(
            getattr(adapter_config, "modules_to_save", None)
        ),
        "bias": normalize_config_value(getattr(adapter_config, "bias", None)),
        "inference_mode": normalize_config_value(
            getattr(adapter_config, "inference_mode", None)
        ),
    }


def build_peft_config_fingerprint(
    peft_config: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "peft_type": normalize_config_value(peft_config.get("peft_type", "LORA")),
        "task_type": normalize_config_value(peft_config.get("task_type")),
        "base_model_name_or_path": None,
        "r": normalize_config_value(peft_config.get("r")),
        "lora_alpha": normalize_config_value(peft_config.get("lora_alpha")),
        "target_modules": normalize_config_value(peft_config.get("target_modules")),
        "modules_to_save": normalize_config_value(peft_config.get("modules_to_save")),
        "bias": normalize_config_value(peft_config.get("bias")),
        "inference_mode": normalize_config_value(peft_config.get("inference_mode")),
    }


def normalize_adapter_path(
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


def build_merged_model_auto_resolution_path(
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


def get_adapter_base_model(
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


def validate_adapter_base_model(
    adapter_config: Any,
    pretrained_model_name: str,
) -> None:
    adapter_base_model = get_adapter_base_model(adapter_config=adapter_config)

    if os.path.normpath(adapter_base_model) != os.path.normpath(pretrained_model_name):
        raise ValueError(
            "Adapter base model mismatch: "
            f"adapter={adapter_base_model}, current={pretrained_model_name}"
        )


def validate_adapter_lora_config(
    adapter_config: Any,
    peft_config_dict: Dict[str, Any],
) -> None:
    validate_equal_config(
        name="peft_type",
        adapter_value=getattr(adapter_config, "peft_type", None),
        expected_value=peft_config_dict.get("peft_type", "LORA"),
    )
    validate_equal_config(
        name="task_type",
        adapter_value=getattr(adapter_config, "task_type", None),
        expected_value=peft_config_dict.get("task_type"),
    )
    validate_equal_int_config(
        name="r",
        adapter_value=getattr(adapter_config, "r", None),
        expected_value=peft_config_dict.get("r"),
    )
    validate_equal_int_config(
        name="lora_alpha",
        adapter_value=getattr(adapter_config, "lora_alpha", None),
        expected_value=peft_config_dict.get("lora_alpha"),
    )
    validate_target_modules(
        adapter_value=getattr(adapter_config, "target_modules", None),
        expected_value=peft_config_dict.get("target_modules"),
    )
    validate_equal_config(
        name="bias",
        adapter_value=getattr(adapter_config, "bias", None),
        expected_value=peft_config_dict.get("bias"),
    )
    validate_equal_config(
        name="modules_to_save",
        adapter_value=getattr(adapter_config, "modules_to_save", None),
        expected_value=peft_config_dict.get("modules_to_save"),
    )


def validate_equal_int_config(
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


def validate_equal_config(
    name: str,
    adapter_value: Any,
    expected_value: Any,
) -> None:
    normalized_adapter_value = normalize_config_value(adapter_value)
    normalized_expected_value = normalize_config_value(expected_value)
    if normalized_adapter_value != normalized_expected_value:
        raise ValueError(
            f"Adapter {name} mismatch: adapter={normalized_adapter_value}, "
            f"expected={normalized_expected_value}"
        )


def validate_target_modules(
    adapter_value: Any,
    expected_value: Any,
) -> None:
    adapter_target_modules = normalize_target_modules(value=adapter_value)
    expected_target_modules = normalize_target_modules(value=expected_value)

    if expected_target_modules == ["all-linear"]:
        if len(adapter_target_modules) == 0:
            raise ValueError("Adapter target_modules is empty.")
        return

    if adapter_target_modules != expected_target_modules:
        raise ValueError(
            "Adapter target_modules mismatch: "
            f"adapter={adapter_target_modules}, expected={expected_target_modules}"
        )


def normalize_target_modules(
    value: Any,
) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return sorted(str(item) for item in value)
    raise TypeError(f"Unsupported target_modules type: {type(value).__name__}")


def normalize_config_value(
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


def validate_trainable_lora_parameters(
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
