import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["HF_HOME"] = os.environ.get("HF_HOME")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from typing import Dict, List, Tuple, Optional
import re
import json

import torch

from transformers import AutoModelForCausalLM

import hydra
from omegaconf import DictConfig


def torch_dtype_from_str(dtype_str: str) -> torch.dtype:
    dtype_str = str(dtype_str).lower()
    if dtype_str in ("fp16", "float16"):
        return torch.float16
    if dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_str in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}")


def _is_expert_ffn_key(key: str) -> bool:
    return (
        re.match(
            r"^model\.layers\.\d+\.mlp\.experts\.\d+\.(up_proj|gate_proj|down_proj)\.weight$",
            key,
        )
        is not None
    )


def _is_attn_key(
    key: str,
    attn_projs: List[str],
) -> bool:
    pat = (
        r"^model\.layers\.\d+\.self_attn\.("
        + "|".join([re.escape(proj) for proj in attn_projs])
        + r")\.weight$"
    )
    return (
        re.match(
            pat,
            key,
        )
        is not None
    )


def _resolve_adapter_for_expert(
    d2m_cfg: DictConfig,
    merge_mode: str,
    expert: int,
    num_experts: int,
) -> Tuple[str, float]:
    if merge_mode == "one2n":
        adapter_dir = str(d2m_cfg.one2n.base_adapter)
        alpha = float(list(d2m_cfg.one2n.alphas)[expert])
        return adapter_dir, alpha
    if merge_mode == "n2n":
        adapter_dir_rel = str(list(d2m_cfg.n2n.adapter_paths)[expert])
        adapter_dir = os.path.join(
            str(d2m_cfg.base_adapter_dir),
            adapter_dir_rel,
        )
        if d2m_cfg.n2n.get("alphas", None) is None:
            alpha = 1.0
        else:
            alpha = float(list(d2m_cfg.n2n.alphas)[expert])
        return adapter_dir, alpha
    if merge_mode == "m2n":
        adapter_dir_rel_list = [str(x) for x in list(d2m_cfg.m2n.adapter_paths)]
        adapter_dir_full = [
            os.path.join(
                str(d2m_cfg.base_adapter_dir),
                adapter_dir_rel,
            )
            for adapter_dir_rel in adapter_dir_rel_list
        ]
        idx = int(list(d2m_cfg.m2n.expert_to_adapter)[expert])
        adapter_dir = adapter_dir_full[idx]
        if d2m_cfg.m2n.get("per_expert_alphas", None) is not None:
            alpha = float(list(d2m_cfg.m2n.per_expert_alphas)[expert])
        else:
            group_alphas = [float(x) for x in list(d2m_cfg.m2n.group_alphas)]
            num_groups = len(group_alphas)
            if d2m_cfg.m2n.get("expert_to_group", None) is not None:
                grp = int(list(d2m_cfg.m2n.expert_to_group)[expert])
            else:
                grp = min((expert * num_groups) // num_experts, num_groups - 1)
            alpha = float(group_alphas[grp])
        return adapter_dir, alpha
    raise ValueError(f"Unknown merge_mode={merge_mode}")


def _get_adapter(
    adapter_cache: Dict[str, Tuple[Dict, Dict[str, torch.Tensor]]],
    adapter_dir: str,
) -> Tuple[Dict, Dict[str, torch.Tensor]]:
    if adapter_dir not in adapter_cache:
        adapter_cache[adapter_dir] = (
            _load_lora_config(adapter_dir=adapter_dir),
            _load_lora_state_dict(adapter_dir=adapter_dir),
        )
    return adapter_cache[adapter_dir]


def _load_lora_config(adapter_dir: str) -> Dict[str, str]:
    cfg_path = os.path.join(
        adapter_dir,
        "adapter_config.json",
    )
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing adapter_config.json under: {adapter_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_lora_state_dict(adapter_dir: str) -> Dict[str, torch.Tensor]:
    safepath = os.path.join(
        adapter_dir,
        "adapter_model.safetensors",
    )
    binpath = os.path.join(
        adapter_dir,
        "adapter_model.bin",
    )
    if os.path.exists(safepath):
        from safetensors.torch import load_file

        return load_file(safepath)
    if os.path.exists(binpath):
        return torch.load(
            binpath,
            map_location="cpu",
        )
    raise FileNotFoundError(
        f"Cannot find adapter_model.safetensors or adapter_model.bin under: {adapter_dir}"
    )


def _list_ffn_lora_targets_from_adapter(
    lora_state_dict: Dict[str, torch.Tensor],
) -> List[Tuple[int, str]]:
    """
    Extract (layer_index, proj_name) pairs for FFN LoRA targets that exist in the adapter.
    This function is intentionally robust to:
    - extra prefixes (e.g., base_model.model.)
    - duplicated 'model.model.' segments
    - adapter-name segments (e.g., .lora_A.default)
    """
    a_keys = [
        k for k in lora_state_dict.keys() if ".lora_A" in k and k.endswith(".weight")
    ]
    b_keys = [
        k for k in lora_state_dict.keys() if ".lora_B" in k and k.endswith(".weight")
    ]

    def _normalize_module_path(key: str) -> str:
        key_without_weight = key[: -len(".weight")]

        key_without_lora = re.sub(
            r"\.lora_[AB](\.[^.]+)?$",
            "",
            key_without_weight,
        )

        if "model." in key_without_lora:
            key_without_lora = key_without_lora[key_without_lora.find("model.") :]

        while key_without_lora.startswith("model.model."):
            key_without_lora = "model." + key_without_lora[len("model.model.") :]

        key_without_lora = key_without_lora.replace(
            "model.model.layers.", "model.layers."
        )

        if key_without_lora.startswith("layers."):
            key_without_lora = "model." + key_without_lora

        return key_without_lora

    a_modules = set(_normalize_module_path(k) for k in a_keys)
    b_modules = set(_normalize_module_path(k) for k in b_keys)
    modules_with_both = sorted(list(a_modules.intersection(b_modules)))

    ffn_targets: List[Tuple[int, str]] = []

    ffn_pattern = re.compile(
        r"^(?:model\.)?layers\.(\d+)\.mlp\.(up_proj|gate_proj|down_proj)$"
    )
    ffn_pattern_model_layers = re.compile(
        r"^model\.layers\.(\d+)\.mlp\.(up_proj|gate_proj|down_proj)$"
    )

    for module_path in modules_with_both:
        match = ffn_pattern_model_layers.match(module_path)
        if match is None:
            match = ffn_pattern.match(module_path.replace("model.", "", 1))
        if match is None:
            continue

        layer_index = int(match.group(1))
        proj_name = str(match.group(2))
        ffn_targets.append(
            (
                layer_index,
                proj_name,
            )
        )

    ffn_targets.sort(key=lambda x: (x[0], x[1]))
    return ffn_targets


def _collect_lora_A_B(
    lora_state_dict: Dict[str, torch.Tensor],
    target_weight_key: str,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Robust matcher for PEFT LoRA keys.
    It supports common PEFT key patterns such as:
    - base_model.model.model.layers.53.mlp.up_proj.lora_A.weight
    - base_model.model.model.layers.53.mlp.up_proj.lora_A.default.weight
    - ...lora_A.<adapter_name>.weight
    It also works when the target module path differs only by additional prefixes.
    """
    target_module_path = target_weight_key[: -len(".weight")]

    target_module_path_tail = target_module_path
    if "model." in target_module_path_tail:
        target_module_path_tail = target_module_path_tail[
            target_module_path_tail.find("model.") :
        ]

    a_candidates: List[str] = []
    b_candidates: List[str] = []

    for key in lora_state_dict.keys():
        if not key.endswith(".weight"):
            continue

        if (".lora_A" not in key) and (".lora_B" not in key):
            continue

        contains_full = target_module_path in key
        contains_tail = target_module_path_tail in key

        if not (contains_full or contains_tail):
            continue

        if (".lora_A." in key) or key.endswith(".lora_A.weight"):
            a_candidates.append(key)
        if (".lora_B." in key) or key.endswith(".lora_B.weight"):
            b_candidates.append(key)

    if not a_candidates or not b_candidates:
        return None

    def _score(candidate_key: str) -> Tuple[int, int, int, int]:
        """
        Higher is better.
        Priority:
        1) prefer explicit '.default.' adapter
        2) prefer exact tail match of module path
        3) prefer keys that end exactly with '.lora_A.weight' / '.lora_B.weight'
        4) prefer shorter keys (fewer prefixes)
        """
        score_default = 1 if ".default." in candidate_key else 0
        score_tail_match = 1 if target_module_path_tail in candidate_key else 0
        score_exact_end = (
            1
            if (
                candidate_key.endswith(".lora_A.weight")
                or candidate_key.endswith(".lora_B.weight")
            )
            else 0
        )
        score_shorter = -len(candidate_key)
        return (
            score_default,
            score_tail_match,
            score_exact_end,
            score_shorter,
        )

    a_key = sorted(
        a_candidates,
        key=_score,
        reverse=True,
    )[0]
    b_key = sorted(
        b_candidates,
        key=_score,
        reverse=True,
    )[0]

    return (
        lora_state_dict[a_key].to(torch.float32),
        lora_state_dict[b_key].to(torch.float32),
    )


def _expected_delta(
    A: torch.Tensor,
    B: torch.Tensor,
    lora_alpha: float,
    r: int,
    alpha_override: float,
    weight_coef: float,
) -> torch.Tensor:
    scale = float(lora_alpha) / float(r)
    return (B @ A) * (scale * float(alpha_override) * float(weight_coef))


def _max_abs_diff(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    return float((actual - expected).abs().max().item())


@hydra.main(
    config_path="../../configs/",
    config_name="sft.yaml",
)
def verify_lora_merge(
    config: DictConfig,
) -> None:
    d2m_cfg = config.dense_to_moe

    base_moe_dir = str(d2m_cfg.moe_model_dir)
    merged_dir = str(d2m_cfg.merged_moe_model_dir)

    dtype = torch_dtype_from_str(dtype_str=d2m_cfg.runtime.dtype)
    device = torch.device(str(d2m_cfg.runtime.device))
    trust_remote_code = bool(d2m_cfg.runtime.trust_remote_code)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_moe_dir,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    ).to(device)
    base_model.eval()
    base_state_dict = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}

    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    ).to(device)
    merged_model.eval()
    merged_state_dict = {
        k: v.detach().cpu() for k, v in merged_model.state_dict().items()
    }

    attn_projs = [str(x) for x in list(d2m_cfg.targets.attn_projs)]
    merge_attn = bool(d2m_cfg.targets.merge_attention)

    changed = []
    unchanged = []
    for k in base_state_dict.keys():
        if k not in merged_state_dict:
            continue
        if torch.equal(base_state_dict[k], merged_state_dict[k]):
            unchanged.append(k)
        else:
            changed.append(k)

    if len(changed) == 0:
        raise RuntimeError(
            "No weights changed between base_moe_dir and merged_dir. Merge likely failed."
        )

    allowed_changed = []
    forbidden_changed = []
    for k in changed:
        ok = _is_expert_ffn_key(key=k)
        if merge_attn:
            ok = ok or _is_attn_key(
                key=k,
                attn_projs=attn_projs,
            )
        if ok:
            allowed_changed.append(k)
        else:
            forbidden_changed.append(k)

    if len(forbidden_changed) > 0:
        sample = forbidden_changed[:50]
        raise RuntimeError(
            f"Found {len(forbidden_changed)} forbidden changed keys. Sample:\n"
            + "\n".join(sample)
        )

    print(
        f"[diff] changed={len(changed)} allowed_changed={len(allowed_changed)} forbidden_changed=0"
    )

    num_experts = int(d2m_cfg.moe.num_experts)
    mode = str(d2m_cfg.merge_mode)

    vcfg = getattr(d2m_cfg, "verify_merge", None)
    if vcfg is None:
        sample_experts = [0, num_experts - 1]
        max_module_samples = 8
        strict_targets = False
        prefer_projs = ["up_proj", "gate_proj", "down_proj"]
    else:
        sample_experts = list(vcfg.experts)
        max_module_samples = int(getattr(vcfg, "max_module_samples", 8))
        strict_targets = bool(getattr(vcfg, "strict_targets", False))
        prefer_projs = list(
            getattr(vcfg, "projs", ["up_proj", "gate_proj", "down_proj"])
        )

    adapter_cache: Dict[str, Tuple[Dict, Dict[str, torch.Tensor]]] = {}

    tol = float(getattr(d2m_cfg, "verify_merge", {}).get("delta_tol", 5e-3))
    tol_attn = float(getattr(d2m_cfg, "verify_merge", {}).get("attn_delta_tol", 5e-3))

    representative_adapter_dir, _ = _resolve_adapter_for_expert(
        d2m_cfg=d2m_cfg,
        merge_mode=mode,
        expert=0,
        num_experts=num_experts,
    )
    _, representative_lora_state_dict = _get_adapter(
        adapter_cache=adapter_cache,
        adapter_dir=representative_adapter_dir,
    )

    ffn_targets_in_adapter = _list_ffn_lora_targets_from_adapter(
        lora_state_dict=representative_lora_state_dict,
    )
    if len(ffn_targets_in_adapter) == 0:
        adapter_key_preview_all = list(representative_lora_state_dict.keys())[:50]
        adapter_key_preview_mlp = [
            k
            for k in representative_lora_state_dict.keys()
            if ("mlp" in k or "feed_forward" in k)
        ][:50]
        raise RuntimeError(
            "No FFN LoRA targets found inside adapter: "
            + f"{representative_adapter_dir}. "
            + "Check PEFT target_modules and adapter contents.\n"
            + "Adapter key preview (first 50 keys):\n"
            + "\n".join(adapter_key_preview_all)
            + "\n\n"
            + "Adapter key preview containing 'mlp' or 'feed_forward' (up to 50):\n"
            + "\n".join(adapter_key_preview_mlp)
        )

    proj_rank = {proj_name: i for i, proj_name in enumerate(prefer_projs)}
    ffn_targets_in_adapter.sort(
        key=lambda x: (
            proj_rank.get(x[1], 999),
            x[0],
        )
    )

    sample_layer_proj_pairs = ffn_targets_in_adapter[:max_module_samples]
    sample_layers = sorted(list(set([layer for layer, _ in sample_layer_proj_pairs])))

    if strict_targets:
        num_hidden_layers = int(getattr(base_model.config, "num_hidden_layers", 0))
        if num_hidden_layers <= 0:
            raise RuntimeError(
                "Could not determine num_hidden_layers from base model config."
            )
        expected_pairs = set(
            (layer_index, proj_name)
            for layer_index in range(num_hidden_layers)
            for proj_name in ["up_proj", "gate_proj", "down_proj"]
        )
        actual_pairs = set(ffn_targets_in_adapter)
        missing_pairs = sorted(list(expected_pairs - actual_pairs))
        if len(missing_pairs) > 0:
            missing_preview = missing_pairs[:50]
            raise RuntimeError(
                "Strict mode: adapter does not contain FFN LoRA targets for all layers/projections. "
                f"Missing sample (up to 50): {missing_preview}"
            )

    delta_checks = []
    for layer, proj in sample_layer_proj_pairs:
        for expert in sample_experts:
            adapter_dir, alpha_override = _resolve_adapter_for_expert(
                d2m_cfg=d2m_cfg,
                merge_mode=mode,
                expert=int(expert),
                num_experts=num_experts,
            )
            lora_cfg, lora_state_dict = _get_adapter(
                adapter_cache=adapter_cache,
                adapter_dir=adapter_dir,
            )
            r = int(lora_cfg["r"])
            lora_alpha = float(lora_cfg["lora_alpha"])

            expert_key = (
                f"model.layers.{int(layer)}.mlp.experts.{int(expert)}.{proj}.weight"
            )
            dense_key = f"model.layers.{int(layer)}.mlp.{proj}.weight"

            A_B = _collect_lora_A_B(
                lora_state_dict=lora_state_dict,
                target_weight_key=dense_key,
            )
            if A_B is None:
                target_module_path = dense_key[: -len(".weight")]
                target_module_path_tail = target_module_path
                if "model." in target_module_path_tail:
                    target_module_path_tail = target_module_path_tail[
                        target_module_path_tail.find("model.") :
                    ]
                candidate_keys = [
                    k
                    for k in lora_state_dict.keys()
                    if (target_module_path in k or target_module_path_tail in k)
                    and ("lora_" in k)
                ]
                candidate_preview = candidate_keys[:30]
                raise RuntimeError(
                    "LoRA A/B not found for target "
                    + f"{dense_key} in adapter {adapter_dir}\n"
                    + "Candidate LoRA keys for this target (up to 30):\n"
                    + "\n".join(candidate_preview)
                )

            A, B = A_B

            expected = _expected_delta(
                A=A,
                B=B,
                lora_alpha=lora_alpha,
                r=r,
                alpha_override=alpha_override,
                weight_coef=1.0,
            )

            base_weight = base_state_dict[expert_key].float()
            merged_weight = merged_state_dict[expert_key].float()
            actual = merged_weight - base_weight

            max_abs_diff = _max_abs_diff(
                actual=actual,
                expected=expected,
            )
            passed = bool(max_abs_diff <= tol)
            delta_checks.append(
                {
                    "layer": int(layer),
                    "expert": int(expert),
                    "proj": proj,
                    "adapter": adapter_dir,
                    "alpha_override": float(alpha_override),
                    "max_abs_diff(actual_vs_expected)": float(max_abs_diff),
                    "pass": passed,
                }
            )
            if not passed:
                raise RuntimeError(
                    f"[FFN delta mismatch] layer={layer} expert={expert} proj={proj} "
                    f"max_abs_diff={max_abs_diff} (tol={tol})"
                )

    print(
        f"[verify_merge] FFN delta checks passed for samples (tol={tol}, samples={len(delta_checks)})."
    )

    if merge_attn:
        attn_alpha = float(d2m_cfg.targets.attn_alpha)

        if mode == "one2n":
            attn_adapters = [
                _resolve_adapter_for_expert(
                    d2m_cfg=d2m_cfg,
                    merge_mode=mode,
                    expert=0,
                    num_experts=num_experts,
                )[0]
            ]
            coefs = [1.0]
        else:
            uniq = sorted(
                set(
                    [
                        _resolve_adapter_for_expert(
                            d2m_cfg=d2m_cfg,
                            merge_mode=mode,
                            expert=expert,
                            num_experts=num_experts,
                        )[0]
                        for expert in range(num_experts)
                    ]
                )
            )
            k = len(uniq)
            attn_adapters = uniq
            coefs = [1.0 / float(k) for _ in range(k)]

        attn_layer = int(sample_layers[0]) if len(sample_layers) > 0 else 0
        attn_proj = str(attn_projs[0])
        base_k = f"model.layers.{attn_layer}.self_attn.{attn_proj}.weight"
        if base_k not in base_state_dict:
            attn_layer = 0
            base_k = f"model.layers.{attn_layer}.self_attn.{attn_proj}.weight"
        if base_k not in base_state_dict:
            raise RuntimeError(f"Attention key not found in model state_dict: {base_k}")

        expected_sum = None
        for adapter_dir, coef in zip(attn_adapters, coefs):
            lora_cfg, lora_state_dict = _get_adapter(
                adapter_cache=adapter_cache,
                adapter_dir=adapter_dir,
            )
            r = int(lora_cfg["r"])
            lora_alpha = float(lora_cfg["lora_alpha"])
            A_B = _collect_lora_A_B(
                lora_state_dict=lora_state_dict,
                target_weight_key=base_k,
            )
            if A_B is None:
                raise RuntimeError(
                    f"LoRA A/B not found for attention target {base_k} in adapter {adapter_dir}"
                )
            A, B = A_B
            d = _expected_delta(
                A=A,
                B=B,
                lora_alpha=lora_alpha,
                r=r,
                alpha_override=attn_alpha,
                weight_coef=coef,
            )
            expected_sum = d if expected_sum is None else (expected_sum + d)

        actual = merged_state_dict[base_k].float() - base_state_dict[base_k].float()
        max_abs_diff = _max_abs_diff(
            actual=actual,
            expected=expected_sum,
        )
        if max_abs_diff > tol_attn:
            raise RuntimeError(
                f"[ATTN delta mismatch] key={base_k} max_abs_diff={max_abs_diff} (tol={tol_attn})"
            )

        print(
            f"[verify_merge] Attention averaging delta check passed (key={base_k}, tol={tol_attn})."
        )

    out_path = os.path.join(merged_dir, "verify_lora_merge_report.json")
    report = {
        "base_moe_dir": base_moe_dir,
        "merged_dir": merged_dir,
        "changed_keys": len(changed),
        "allowed_changed_keys": len(allowed_changed),
        "merge_attention": merge_attn,
        "attn_projs": attn_projs,
        "sample_layer_proj_pairs": sample_layer_proj_pairs,
        "delta_checks": delta_checks,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[OK] LoRA merge verification report saved to: {out_path}")


if __name__ == "__main__":
    verify_lora_merge()
