from typing import Dict, Optional, Any

from omegaconf import DictConfig


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


def patch_grpo_vllm_mm_processor_kwargs(
    mm_processor_kwargs: Dict[str, Any],
) -> None:
    try:
        from trl.trainer import grpo_trainer
    except Exception:
        return

    if getattr(grpo_trainer, "_vllm_mm_processor_patched", False):
        return

    original_llm = grpo_trainer.LLM

    def llm_wrapper(*args, **kwargs):
        if "mm_processor_kwargs" not in kwargs:
            kwargs["mm_processor_kwargs"] = mm_processor_kwargs
        return original_llm(*args, **kwargs)

    grpo_trainer.LLM = llm_wrapper
    grpo_trainer._vllm_mm_processor_patched = True
