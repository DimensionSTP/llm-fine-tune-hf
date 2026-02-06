from typing import Dict, Optional, Any

from omegaconf import DictConfig, OmegaConf


def get_vllm_mm_processor_kwargs(
    config: DictConfig,
    data_encoder,
) -> Optional[Dict[str, Any]]:
    if config.modality == "text":
        return None

    if not config.enable_vllm_mm_processor_kwargs:
        return None

    max_pixels = config.max_pixels
    if max_pixels is None:
        image_processor = data_encoder.image_processor
        max_pixels = image_processor.max_pixels

    if max_pixels is None:
        return None

    return {"max_pixels": int(max_pixels)}


def get_vllm_hf_overrides(
    config: DictConfig,
) -> Optional[Dict[str, Any]]:
    if not config.enable_vllm_hf_overrides:
        return None

    overrides = config.vllm_hf_overrides
    if overrides is None:
        return None

    if isinstance(overrides, DictConfig):
        overrides = OmegaConf.to_container(
            overrides,
            resolve=True,
        )

    if not overrides:
        return None

    return overrides


def patch_grpo_vllm_kwargs(
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    hf_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        from trl.trainer import grpo_trainer
    except Exception:
        return

    if mm_processor_kwargs is None and hf_overrides is None:
        return

    if getattr(grpo_trainer, "_vllm_mm_processor_patched", False):
        return

    original_llm = grpo_trainer.LLM

    def llm_wrapper(*args, **kwargs):
        if mm_processor_kwargs is not None and "mm_processor_kwargs" not in kwargs:
            kwargs["mm_processor_kwargs"] = mm_processor_kwargs
        if hf_overrides is not None and "hf_overrides" not in kwargs:
            kwargs["hf_overrides"] = hf_overrides
        return original_llm(*args, **kwargs)

    grpo_trainer.LLM = llm_wrapper
    grpo_trainer._vllm_mm_processor_patched = True
