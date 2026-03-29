from typing import Callable, Any
from contextlib import nullcontext
from types import MethodType

import torch

from omegaconf import DictConfig


def _build_router_with_lora_sync(
    *,
    remap_name: Callable[[str], str],
) -> Callable[[Any], None]:
    def sync_weights_router_with_lora(
        self: Any,
    ) -> None:
        if self.mode == "colocate" and self.enable_sleep_mode:
            torch.cuda.empty_cache()
            self.llm.wake_up(tags=["weights"])

        model = self.model
        accelerator = self.accelerator
        deepspeed_plugin = accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

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
                    self.vllm_client.update_named_param(name, param.data)
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

    return sync_weights_router_with_lora


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


def patch_qwen_packed_moe_vllm_sync(
    trainer: Any,
    config: DictConfig,
) -> bool:
    """Patch vLLM sync for sparse Qwen checkpoints whose experts stay under mlp.*."""
    is_qwen_packed_moe = config.model_type.startswith("Qwen3-") and (
        "-experts_" in config.model_type
    )
    should_patch = (
        config.fine_tune_method == "grpo"
        and config.use_vllm
        and config.is_peft
        and config.dense_to_moe.router_with_lora
        and is_qwen_packed_moe
        and hasattr(trainer, "vllm_generation")
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
    """Patch vLLM sync for sparse decoder checkpoints remapped to Mixtral-style modules."""
    is_sparse_decoder_moe = (
        "-experts_" in config.model_type and not config.model_type.startswith("Qwen3-")
    )
    should_patch = (
        config.fine_tune_method == "grpo"
        and config.use_vllm
        and config.is_peft
        and config.dense_to_moe.router_with_lora
        and is_sparse_decoder_moe
        and hasattr(trainer, "vllm_generation")
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
