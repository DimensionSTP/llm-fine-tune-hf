from typing import Dict, List, Callable, Any
import os
from contextlib import nullcontext
from fnmatch import fnmatchcase
from functools import partial
from importlib.metadata import version
from types import MethodType

from omegaconf import DictConfig, ListConfig

from packaging.specifiers import SpecifierSet

import torch

import bitsandbytes as bnb

import deepspeed


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


def resolve_lora_streaming_name_remap_config(
    config: DictConfig,
) -> Dict[str, Any]:
    should_resolve = config.fine_tune_method == "distillation" or (
        config.fine_tune_method == "grpo" and config.is_peft
    )
    if not should_resolve or not config.use_vllm:
        return {}

    remap_config = config.vllm_lora_name_remap
    runtime_package_versions = {
        str(package_name): version(str(package_name))
        for package_name in remap_config.version_packages
    }
    resolved = _resolve_lora_streaming_name_remap_profile(
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


def patch_vllm_param_name_remap(
    trainer: Any,
    config: DictConfig,
) -> bool:
    should_patch = (
        config.use_vllm
        and hasattr(
            trainer,
            "vllm_generation",
        )
        and not hasattr(
            trainer.vllm_generation,
            "_push_param_to_vllm_original",
        )
        and (
            config.fine_tune_method == "distillation"
            or (
                config.fine_tune_method == "grpo"
                and config.is_peft
                and config.vllm_sync_strategy == "default"
            )
        )
    )
    if not should_patch:
        return False

    resolve_lora_streaming_name_remap_config(config=config)
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

    resolve_lora_streaming_name_remap_config(config=config)
    trainer.vllm_generation._lora_streaming_name_remapper = (
        _get_lora_streaming_name_remapper(config=config)
    )
    trainer.vllm_generation.sync_weights = MethodType(
        _sync_weights_lora_streaming,
        trainer.vllm_generation,
    )
    return True


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


def _remap_lora_streaming_name(
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
        _remap_lora_streaming_name,
        prefix_rules=profile.prefix_rules,
    )


def _selector_matches_lora_streaming_runtime(
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


def _resolve_lora_streaming_name_remap_profile(
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
        if _selector_matches_lora_streaming_runtime(
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
