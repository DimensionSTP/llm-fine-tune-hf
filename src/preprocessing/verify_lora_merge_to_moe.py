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


def _collect_lora_A_B(
    lora_state_dict: Dict[str, torch.Tensor],
    target_weight_key: str,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Robust matcher for PEFT LoRA keys.

    PEFT/TRL adapters often store keys with extra prefixes and/or adapter names, e.g.
    - base_model.model.model.layers.53.mlp.up_proj.lora_A.weight
    - base_model.model.model.layers.53.mlp.up_proj.lora_A.default.weight
    - ...lora_A.<adapter_name>.weight

    This function matches by:
    1) requiring the base module path (target_path) to appear as a substring
    2) requiring .lora_A / .lora_B to appear
    3) requiring the key to end with ".weight"
    Then chooses the "best" candidate if multiple exist.
    """
    target_path = target_weight_key[
        : -len(".weight")
    ]  # e.g., model.layers.53.mlp.up_proj

    a_candidates: List[str] = []
    b_candidates: List[str] = []

    for k in lora_state_dict.keys():
        if not k.endswith(".weight"):
            continue
        # allow arbitrary prefixes before target_path
        if target_path not in k:
            continue
        # accept both ".lora_A.weight" and ".lora_A.<name>.weight"
        if (".lora_A." in k) or k.endswith(".lora_A.weight"):
            a_candidates.append(k)
        if (".lora_B." in k) or k.endswith(".lora_B.weight"):
            b_candidates.append(k)

    if not a_candidates or not b_candidates:
        return None

    def _score(key: str) -> Tuple[int, int, int]:
        """
        Higher is better.
        Priority:
        1) prefer explicit '.default.' adapter
        2) prefer shorter keys (fewer prefixes)
        3) prefer keys that end exactly with '.lora_A.weight' / '.lora_B.weight'
        """
        s_default = 1 if ".default." in key else 0
        s_exact = (
            1
            if (key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight"))
            else 0
        )
        s_short = -len(key)
        return (s_default, s_exact, s_short)

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
        sample_layers = [0, 1]
        sample_experts = [0, num_experts - 1]
        sample_projs = ["up_proj"]
    else:
        sample_layers = list(vcfg.layers)
        sample_experts = list(vcfg.experts)
        sample_projs = list(vcfg.projs)

    adapter_cache: Dict[str, Tuple[Dict, Dict[str, torch.Tensor]]] = {}

    tol = float(getattr(d2m_cfg, "verify_merge", {}).get("delta_tol", 5e-3))

    delta_checks = []
    for layer in sample_layers:
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

            for proj in sample_projs:
                expert_key = (
                    f"model.layers.{int(layer)}.mlp.experts.{int(expert)}.{proj}.weight"
                )
                dense_key = f"model.layers.{int(layer)}.mlp.{proj}.weight"

                A_B = _collect_lora_A_B(
                    lora_state_dict=lora_state_dict,
                    target_weight_key=dense_key,
                )
                if A_B is None:
                    raise RuntimeError(
                        f"LoRA A/B not found for target {dense_key} in adapter {adapter_dir}"
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
                delta_checks.append(
                    {
                        "layer": int(layer),
                        "expert": int(expert),
                        "proj": proj,
                        "adapter": adapter_dir,
                        "alpha_override": float(alpha_override),
                        "max_abs_diff(actual_vs_expected)": float(max_abs_diff),
                        "pass": bool(max_abs_diff <= tol),
                    }
                )
                if max_abs_diff > tol:
                    raise RuntimeError(
                        f"[FFN delta mismatch] layer={layer} expert={expert} proj={proj} "
                        f"max_abs_diff={max_abs_diff} (tol={tol})"
                    )

    print(f"[verify_merge] FFN delta checks passed for samples (tol={tol}).")

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

        attn_layer = int(sample_layers[0])
        attn_proj = str(attn_projs[0])
        base_k = f"model.layers.{attn_layer}.self_attn.{attn_proj}.weight"

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
        tol_attn = float(
            getattr(d2m_cfg, "verify_merge", {}).get("attn_delta_tol", 5e-3)
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
