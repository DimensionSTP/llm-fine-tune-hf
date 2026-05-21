from typing import Any
from types import MethodType

from omegaconf import DictConfig, ListConfig

import torch


def patch_grpo_completion_termination(
    trainer: Any,
    config: DictConfig,
) -> bool:
    if config.fine_tune_method != "grpo":
        return False

    if not hasattr(config, "completion_termination"):
        return False

    termination_config = config.completion_termination
    if not bool(termination_config.enabled):
        return False

    if not hasattr(trainer, "_generate") or not hasattr(
        trainer,
        "_generate_and_score_completions",
    ):
        return False

    if hasattr(trainer, "_original_generate_for_completion_termination"):
        return False

    terminal_token_ids, terminal_token_sequences = _resolve_terminal_tokens(
        trainer=trainer,
        termination_config=termination_config,
    )
    infer_short_completion = bool(
        termination_config.infer_finished_from_short_completion
    )
    max_completion_length = int(trainer.max_completion_length)

    trainer._original_generate_for_completion_termination = trainer._generate
    trainer._original_generate_and_score_for_completion_termination = (
        trainer._generate_and_score_completions
    )
    trainer._completion_termination_terminal_token_ids = terminal_token_ids
    trainer._completion_termination_terminal_token_sequences = terminal_token_sequences
    trainer._completion_termination_infer_short_completion = infer_short_completion
    trainer._completion_termination_max_completion_length = max_completion_length
    trainer._generate = MethodType(
        _generate_with_completion_termination,
        trainer,
    )
    trainer._generate_and_score_completions = MethodType(
        _generate_and_score_with_completion_termination,
        trainer,
    )
    return True


def _resolve_terminal_tokens(
    trainer: Any,
    termination_config: DictConfig,
) -> tuple[set[int], list[tuple[int, ...]]]:
    terminal_token_ids = set()
    tokenizer = trainer._tokenizer

    for token_id in _to_list(termination_config.terminal_token_ids):
        terminal_token_ids.add(int(token_id))

    if bool(termination_config.include_model_generation_eos):
        model = trainer.accelerator.unwrap_model(trainer.model)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            eos_token_id = getattr(generation_config, "eos_token_id", None)
            for token_id in _to_list(eos_token_id):
                terminal_token_ids.add(int(token_id))

    if tokenizer.eos_token_id is not None:
        terminal_token_ids.add(int(tokenizer.eos_token_id))
    if tokenizer.pad_token_id is not None:
        terminal_token_ids.add(int(tokenizer.pad_token_id))

    terminal_token_sequences = []
    for token_text in _to_list(termination_config.terminal_token_texts):
        token_id = tokenizer.convert_tokens_to_ids(token_text)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            terminal_token_ids.add(int(token_id))
            continue

        token_sequence = tokenizer.encode(
            token_text,
            add_special_tokens=False,
        )
        if len(token_sequence) == 1:
            terminal_token_ids.add(int(token_sequence[0]))
        elif len(token_sequence) > 1:
            terminal_token_sequences.append(tuple(int(item) for item in token_sequence))

    return terminal_token_ids, terminal_token_sequences


def _to_list(
    value: Any,
) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return [value]


def _generate_with_completion_termination(
    trainer: Any,
    prompts: list,
):
    output = trainer._original_generate_for_completion_termination(prompts)
    completion_ids = output[1]
    tool_mask = output[2]
    _log_completion_termination_metrics(
        trainer=trainer,
        completion_ids=completion_ids,
        tool_mask=tool_mask,
    )
    return output


def _generate_and_score_with_completion_termination(
    trainer: Any,
    inputs: list[dict[str, torch.Tensor | Any]],
) -> dict[str, torch.Tensor | Any]:
    if not trainer.mask_truncated_completions:
        return trainer._original_generate_and_score_for_completion_termination(inputs)

    trainer.mask_truncated_completions = False
    try:
        output = trainer._original_generate_and_score_for_completion_termination(inputs)
    finally:
        trainer.mask_truncated_completions = True

    completion_ids_list = _unpadded_completion_ids(
        completion_ids=output["completion_ids"],
        completion_mask=output["completion_mask"],
    )
    is_truncated = _build_completion_termination_tensor(
        trainer=trainer,
        completion_ids=completion_ids_list,
        device=output["completion_mask"].device,
    )
    keep_mask = (
        (~is_truncated)
        .unsqueeze(1)
        .to(
            dtype=output["completion_mask"].dtype,
        )
    )
    output["completion_mask"] = output["completion_mask"] * keep_mask
    if "tool_mask" in output:
        output["tool_mask"] = output["tool_mask"] * keep_mask
    return output


def _log_completion_termination_metrics(
    trainer: Any,
    completion_ids: list[list[int]],
    tool_mask: list[list[int]] | None,
) -> None:
    mode = "train" if trainer.model.training else "eval"
    device = trainer.accelerator.device

    if tool_mask is not None:
        completion_lengths = torch.tensor(
            [sum(mask) for mask in tool_mask],
            device=device,
        )
    else:
        completion_lengths = torch.tensor(
            [len(ids) for ids in completion_ids],
            device=device,
        )
    agg_completion_lengths = trainer.accelerator.gather(completion_lengths)
    is_truncated = _build_completion_termination_tensor(
        trainer=trainer,
        completion_ids=completion_ids,
        device=device,
    )
    agg_is_truncated = trainer.accelerator.gather(is_truncated)
    metrics = trainer._metrics[mode]
    metrics["completions/clipped_ratio"][-1] = agg_is_truncated.float().mean().item()
    term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
    if len(term_completion_lengths) == 0:
        term_completion_lengths = torch.zeros(1, device=device)
    metrics["completions/mean_terminated_length"][-1] = (
        term_completion_lengths.float().mean().item()
    )
    metrics["completions/min_terminated_length"][-1] = (
        term_completion_lengths.float().min().item()
    )
    metrics["completions/max_terminated_length"][-1] = (
        term_completion_lengths.float().max().item()
    )


def _build_completion_termination_tensor(
    trainer: Any,
    completion_ids: list[list[int]],
    device: torch.device,
) -> torch.Tensor:
    return _build_truncation_tensor(
        completion_ids=completion_ids,
        device=device,
        terminal_token_ids=trainer._completion_termination_terminal_token_ids,
        terminal_token_sequences=trainer._completion_termination_terminal_token_sequences,
        infer_short_completion=trainer._completion_termination_infer_short_completion,
        max_completion_length=trainer._completion_termination_max_completion_length,
    )


def _build_truncation_tensor(
    completion_ids: list[list[int]],
    device: torch.device,
    terminal_token_ids: set[int],
    terminal_token_sequences: list[tuple[int, ...]],
    infer_short_completion: bool,
    max_completion_length: int,
) -> torch.Tensor:
    return torch.tensor(
        [
            not _is_terminated(
                token_ids=token_ids,
                terminal_token_ids=terminal_token_ids,
                terminal_token_sequences=terminal_token_sequences,
                infer_short_completion=infer_short_completion,
                max_completion_length=max_completion_length,
            )
            for token_ids in completion_ids
        ],
        device=device,
    )


def _is_terminated(
    token_ids: list[int],
    terminal_token_ids: set[int],
    terminal_token_sequences: list[tuple[int, ...]],
    infer_short_completion: bool,
    max_completion_length: int,
) -> bool:
    if infer_short_completion and len(token_ids) < max_completion_length:
        return True

    if not token_ids:
        return False

    if int(token_ids[-1]) in terminal_token_ids:
        return True

    for token_sequence in terminal_token_sequences:
        sequence_length = len(token_sequence)
        if sequence_length <= len(token_ids) and (
            tuple(token_ids[-sequence_length:]) == token_sequence
        ):
            return True

    return False


def _unpadded_completion_ids(
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> list[list[int]]:
    result = []
    for token_ids, mask in zip(completion_ids, completion_mask, strict=True):
        length = int(mask.sum().item())
        result.append([int(token_id) for token_id in token_ids[:length].tolist()])
    return result
