from typing import Any

from omegaconf import DictConfig, OmegaConf

from ..helpers import resolve_effective_dataset_name


def register_hydra_resolvers() -> None:
    OmegaConf.register_new_resolver(
        "reward_save_suffix",
        _reward_save_suffix,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "dataset_effective_name",
        resolve_effective_dataset_name,
        replace=True,
    )


def _to_float(
    value: Any,
    default: float,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_active_reward(
    weight: Any,
) -> bool:
    return (
        _to_float(
            value=weight,
            default=0.0,
        )
        > 0.0
    )


def _reward_save_suffix(
    reward: Any,
) -> str:
    if not isinstance(reward, (dict, DictConfig)):
        return ""

    reward_weight = reward.get("weight", {})
    if not isinstance(reward_weight, (dict, DictConfig)):
        return ""

    parts = [
        str(reward_key)
        for reward_key, weight in reward_weight.items()
        if _is_active_reward(weight)
    ]

    if len(parts) == 0:
        return ""
    return "-rw_" + "+".join(parts)
