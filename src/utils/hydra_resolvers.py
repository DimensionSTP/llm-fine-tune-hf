from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


def _to_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_active_reward(
    weight: Any,
) -> bool:
    return _to_float(weight) > 0.0


def _fmt_float(
    value: Any,
) -> str:
    numeric = _to_float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def _extract_stage_ks(
    stages: Any,
) -> str:
    if not isinstance(stages, (list, ListConfig)):
        return "na"
    ks = []
    for stage in stages:
        if isinstance(stage, (dict, DictConfig)) and "k" in stage:
            ks.append(str(stage["k"]))
    if len(ks) == 0:
        return "na"
    return "_".join(ks)


def _extract_top_ks(
    top_ks: Any,
) -> str:
    if not isinstance(top_ks, (list, ListConfig)):
        return "na"
    values = [str(k) for k in top_ks]
    if len(values) == 0:
        return "na"
    return "_".join(values)


def _reward_save_suffix(
    reward: Any,
) -> str:
    if not isinstance(reward, (dict, DictConfig)):
        return ""

    reward_weight = reward.get("weight", {})
    if not isinstance(reward_weight, (dict, DictConfig)):
        return ""

    parts = []

    if _is_active_reward(reward_weight.get("match")):
        match_cfg = reward.get("match", {})
        incorrect_penalty = match_cfg.get("incorrect_penalty", 0.0) if isinstance(
            match_cfg, (dict, DictConfig)
        ) else 0.0
        if _to_float(incorrect_penalty) > 0.0:
            parts.append(f"match-neg{_fmt_float(incorrect_penalty)}")

    if _is_active_reward(reward_weight.get("code_execution")):
        code_cfg = reward.get("code_execution", {})
        timeout = code_cfg.get("timeout", "na") if isinstance(
            code_cfg, (dict, DictConfig)
        ) else "na"
        parts.append(f"code-t{timeout}")
        wrong_output_penalty = (
            code_cfg.get("wrong_output_penalty", 0.0)
            if isinstance(code_cfg, (dict, DictConfig))
            else 0.0
        )
        non_executable_penalty = (
            code_cfg.get("non_executable_penalty", 0.0)
            if isinstance(code_cfg, (dict, DictConfig))
            else 0.0
        )
        if (
            _to_float(wrong_output_penalty) > 0.0
            or _to_float(non_executable_penalty) > 0.0
        ):
            parts.append(
                f"codeneg-w{_fmt_float(wrong_output_penalty)}-x{_fmt_float(non_executable_penalty)}"
            )

    if _is_active_reward(reward_weight.get("rouge")):
        rouge_cfg = reward.get("rouge", {})
        rouge_type = rouge_cfg.get("type", "na") if isinstance(
            rouge_cfg, (dict, DictConfig)
        ) else "na"
        parts.append(f"rouge-{rouge_type}")

    if _is_active_reward(reward_weight.get("retrieval_hit")):
        retrieval_cfg = reward.get("retrieval", {})
        hit_cfg = retrieval_cfg.get("hit", {}) if isinstance(
            retrieval_cfg, (dict, DictConfig)
        ) else {}
        top_k = hit_cfg.get("top_k", "na")
        stage_ks = _extract_stage_ks(hit_cfg.get("stages"))
        parts.append(f"rhit-k{top_k}-st{stage_ks}")

    if _is_active_reward(reward_weight.get("retrieval_ndcg")):
        retrieval_cfg = reward.get("retrieval", {})
        ndcg_cfg = retrieval_cfg.get("ndcg", {}) if isinstance(
            retrieval_cfg, (dict, DictConfig)
        ) else {}
        reward_mode = ndcg_cfg.get("reward_mode", "na")
        weighting_mode = ndcg_cfg.get("weighting_mode", "na")
        ks = _extract_top_ks(ndcg_cfg.get("top_ks"))
        parts.append(f"rndcg-{reward_mode}-{weighting_mode}-k{ks}")

    if _is_active_reward(reward_weight.get("single_kv")):
        single_kv_cfg = reward.get("single_kv", {})
        json_parse_weight = single_kv_cfg.get("json_parse_weight", "na") if isinstance(
            single_kv_cfg, (dict, DictConfig)
        ) else "na"
        parts.append(f"singlekv-jp{_fmt_float(json_parse_weight)}")

    if _is_active_reward(reward_weight.get("multi_kv")):
        multi_kv_cfg = reward.get("multi_kv", {})
        json_parse_weight = multi_kv_cfg.get("json_parse_weight", "na") if isinstance(
            multi_kv_cfg, (dict, DictConfig)
        ) else "na"
        parts.append(f"multikv-jp{_fmt_float(json_parse_weight)}")

    if len(parts) == 0:
        return ""
    return "-rw_" + "+".join(parts)


def register_hydra_resolvers() -> None:
    OmegaConf.register_new_resolver(
        "reward_save_suffix",
        _reward_save_suffix,
        replace=True,
    )
