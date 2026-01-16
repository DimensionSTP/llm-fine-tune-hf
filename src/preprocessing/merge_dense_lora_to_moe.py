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

from typing import Dict, List, Tuple, Optional, Any

import re
import json
from dataclasses import dataclass

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

import hydra
from omegaconf import DictConfig, OmegaConf


def torch_dtype_from_str(dtype_str: str) -> torch.dtype:
    dtype_str = str(dtype_str).lower()
    if dtype_str in ("fp16", "float16"):
        return torch.float16
    if dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_str in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}")


def _list_expert_ffn_weight_keys(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    pat = re.compile(
        r"^model\.layers\.(\d+)\.mlp\.(?:experts|expert)\.(\d+)\.(up_proj|gate_proj|down_proj)\.weight$"
    )
    return [k for k in state_dict.keys() if pat.match(k)]


def _infer_num_layers_and_experts_from_keys(keys: List[str]) -> Tuple[int, int]:
    layer_max = -1
    expert_max = -1
    pat = re.compile(r"^model\.layers\.(\d+)\.mlp\.(?:experts|expert)\.(\d+)\.")
    for k in keys:
        m = pat.match(k)
        if not m:
            continue
        layer_max = max(layer_max, int(m.group(1)))
        expert_max = max(expert_max, int(m.group(2)))
    if layer_max < 0 or expert_max < 0:
        raise RuntimeError("Could not infer layers/experts from expert FFN keys.")
    return layer_max + 1, expert_max + 1


def _validate_list_length(
    name: str,
    xs: List[Any],
    expected: int,
) -> None:
    if len(xs) != expected:
        raise ValueError(f"{name} must have length {expected}, got {len(xs)}")


@dataclass
class LoraWeights:
    A: torch.Tensor  # [r, in]
    B: torch.Tensor  # [out, r]
    scale: float  # (lora_alpha / r) from adapter config


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


def _load_lora_config(adapter_dir: str) -> Dict:
    cfg_path = os.path.join(
        adapter_dir,
        "adapter_config.json",
    )
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing adapter_config.json under: {adapter_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_lora_for_target(
    lora_state_dict: Dict[str, torch.Tensor],
    adapter_cfg: Dict,
    target_weight_key: str,
) -> Optional[LoraWeights]:
    """
    target_weight_key example:
      model.layers.0.mlp.up_proj.weight
      model.layers.0.self_attn.q_proj.weight
    """
    r = adapter_cfg.get(
        "r",
        None,
    )
    lora_alpha = adapter_cfg.get(
        "lora_alpha",
        None,
    )
    if r is None or lora_alpha is None:
        raise ValueError("adapter_config.json missing 'r' or 'lora_alpha' (top-level).")

    scale = float(lora_alpha) / float(r)
    target_path = target_weight_key[: -len(".weight")]

    a_suffix = f"{target_path}.lora_A.weight"
    b_suffix = f"{target_path}.lora_B.weight"

    A_key = None
    B_key = None
    for k in lora_state_dict.keys():
        if k.endswith(a_suffix):
            A_key = k
        elif k.endswith(b_suffix):
            B_key = k

    if A_key is None or B_key is None:
        return None

    A = lora_state_dict[A_key].to(torch.float32)
    B = lora_state_dict[B_key].to(torch.float32)
    return LoraWeights(
        A=A,
        B=B,
        scale=scale,
    )


def _apply_lora_delta_inplace(
    base_weight: torch.Tensor,
    lora: LoraWeights,
    alpha_override: float,
    weight_coef: float = 1.0,
) -> None:
    """
    base_weight += weight_coef * alpha_override * (B @ A) * (lora.scale)
    - alpha_override: your experiment-level scaling (per-expert for FFN, optional for attention)
    - weight_coef: used for equal-weight averaging across multiple adapters (e.g., 1/4, 1/8).
    """
    delta = (lora.B @ lora.A) * (
        lora.scale * float(alpha_override) * float(weight_coef)
    )
    base_weight.add_(
        delta.to(
            dtype=base_weight.dtype,
            device=base_weight.device,
        )
    )


def _resolve_adapter_plan(
    config: DictConfig,
) -> Tuple[
    List[str],
    List[float],
    Dict[str, Dict],
    Dict[str, Dict[str, torch.Tensor]],
    List[str],
]:
    """
    Returns:
      adapter_for_expert: length n, adapter path per expert (for FFN)
      alpha_for_expert: length n, alpha per expert (for FFN)
      lora_cfg_for_adapter: dict adapter_path -> adapter_config
      lora_state_dict_for_adapter: dict adapter_path -> adapter_state_dict
      all_adapter_paths: list of unique adapter paths involved (for attention averaging)
    """
    num_experts = int(config.moe.num_experts)
    mode = str(config.merge_mode)

    if mode == "one2n":
        base_adapter = str(config.one2n.base_adapter)
        alphas = [float(x) for x in list(config.one2n.alphas)]
        _validate_list_length(
            name="one2n.alphas",
            xs=alphas,
            expected=num_experts,
        )

        adapter_for_expert = [base_adapter for _ in range(num_experts)]
        alpha_for_expert = alphas

        lora_cfg = _load_lora_config(adapter_dir=base_adapter)
        lora_state_dict = _load_lora_state_dict(adapter_dir=base_adapter)

        lora_cfg_for_adapter = {base_adapter: lora_cfg}
        lora_state_dict_for_adapter = {base_adapter: lora_state_dict}
        all_adapter_paths = [base_adapter]

    elif mode == "n2n":
        adapter_paths = [str(p) for p in list(config.n2n.adapter_paths)]
        _validate_list_length(
            name="n2n.adapter_paths",
            xs=adapter_paths,
            expected=num_experts,
        )
        full_adapter_paths = [
            os.path.join(
                config.base_adapter_dir,
                adapter_path,
            )
            for adapter_path in adapter_paths
        ]
        adapter_for_expert = full_adapter_paths

        if config.n2n.get("alphas", None) is None:
            alpha_for_expert = [1.0 for _ in range(num_experts)]
        else:
            alphas = [float(x) for x in list(config.n2n.alphas)]
            _validate_list_length(
                name="n2n.alphas",
                xs=alphas,
                expected=num_experts,
            )
            alpha_for_expert = alphas

        uniq = sorted(set(adapter_for_expert))
        lora_cfg_for_adapter = {ap: _load_lora_config(ap) for ap in uniq}
        lora_state_dict_for_adapter = {ap: _load_lora_state_dict(ap) for ap in uniq}
        all_adapter_paths = uniq

    elif mode == "m2n":
        adapter_paths = [str(p) for p in list(config.m2n.adapter_paths)]
        m = len(adapter_paths)
        if m <= 0:
            raise ValueError("m2n requires m2n.adapter_paths (length m > 0).")

        expert_to_adapter = [int(x) for x in list(config.m2n.expert_to_adapter)]
        _validate_list_length(
            name="m2n.expert_to_adapter",
            xs=expert_to_adapter,
            expected=num_experts,
        )
        if any(x < 0 or x >= m for x in expert_to_adapter):
            raise ValueError(f"m2n.expert_to_adapter must be in [0, {m-1}]")

        full_adapter_paths = [
            os.path.join(
                config.base_adapter_dir,
                adapter_path,
            )
            for adapter_path in adapter_paths
        ]
        adapter_for_expert = [full_adapter_paths[idx] for idx in expert_to_adapter]

        if config.m2n.get("per_expert_alphas", None) is not None:
            per_expert_alphas = [float(x) for x in list(config.m2n.per_expert_alphas)]
            _validate_list_length(
                name="m2n.per_expert_alphas",
                xs=per_expert_alphas,
                expected=num_experts,
            )
            alpha_for_expert = per_expert_alphas
        else:
            group_alphas = [float(x) for x in list(config.m2n.group_alphas)]
            g = len(group_alphas)
            if g <= 0:
                raise ValueError(
                    "m2n requires either lora.per_expert_alphas or lora.group_alphas (len>0)."
                )

            if config.m2n.get("expert_to_group", None) is not None:
                expert_to_group = [int(x) for x in list(config.m2n.expert_to_group)]
                _validate_list_length(
                    name="m2n.expert_to_group",
                    xs=expert_to_group,
                    expected=num_experts,
                )
                if any(x < 0 or x >= g for x in expert_to_group):
                    raise ValueError(f"m2n.expert_to_group must be in [0, {g-1}]")
            else:
                expert_to_group = [
                    min((e * g) // num_experts, g - 1) for e in range(num_experts)
                ]

            alpha_for_expert = [
                group_alphas[expert_to_group[e]] for e in range(num_experts)
            ]

        uniq = sorted(set(adapter_for_expert))
        lora_cfg_for_adapter = {ap: _load_lora_config(ap) for ap in uniq}
        lora_state_dict_for_adapter = {ap: _load_lora_state_dict(ap) for ap in uniq}
        all_adapter_paths = uniq

    else:
        raise ValueError("mode must be one of: one2n | n2n | m2n")

    return (
        adapter_for_expert,
        alpha_for_expert,
        lora_cfg_for_adapter,
        lora_state_dict_for_adapter,
        all_adapter_paths,
    )


@hydra.main(
    config_path="../../configs/",
    config_name="sft.yaml",
)
def merge_dense_lora_to_moe(
    config: DictConfig,
) -> None:
    d2m_cfg = config.dense_to_moe

    moe_model_dir = str(d2m_cfg.moe_model_dir)
    output_dir = str(d2m_cfg.output_dir)
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    dtype = torch_dtype_from_str(dtype_str=d2m_cfg.runtime.dtype)
    device = torch.device(str(d2m_cfg.runtime.device))
    trust_remote_code = bool(d2m_cfg.runtime.trust_remote_code)

    moe_model = AutoModelForCausalLM.from_pretrained(
        moe_model_dir,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    ).to(device)
    moe_model.eval()
    state_dict = moe_model.state_dict()

    expert_keys = _list_expert_ffn_weight_keys(state_dict)
    if not expert_keys:
        raise RuntimeError(
            "No expert FFN keys found. This does not look like a MoE checkpoint with FFN experts."
        )
    num_layers, num_experts_found = _infer_num_layers_and_experts_from_keys(expert_keys)

    num_experts_cfg = int(d2m_cfg.moe.num_experts)
    if num_experts_found != num_experts_cfg:
        raise ValueError(
            f"num_experts mismatch: config.dense_to_moe.moe.num_experts={num_experts_cfg} but checkpoint has {num_experts_found}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        moe_model_dir,
        trust_remote_code=trust_remote_code,
    )

    (
        adapter_for_expert,
        alpha_for_expert,
        lora_cfg_for_adapter,
        lora_state_dict_for_adapter,
        all_adapter_paths,
    ) = _resolve_adapter_plan(config=d2m_cfg)

    ffn_projs = [
        "up_proj",
        "gate_proj",
        "down_proj",
    ]
    attn_projs = [str(x) for x in list(d2m_cfg.targets.attn_projs)]
    merge_attn = bool(d2m_cfg.targets.merge_attention)

    mode = str(d2m_cfg.merge_mode)
    attn_alpha = float(d2m_cfg.targets.attn_alpha)

    if merge_attn:
        if mode == "one2n":
            attn_adapter_paths = all_adapter_paths
            attn_weight_coefs = [1.0]
        else:
            k = len(all_adapter_paths)
            if k <= 0:
                raise RuntimeError("No adapters found for attention averaging.")
            attn_adapter_paths = all_adapter_paths
            attn_weight_coefs = [1.0 / float(k) for _ in range(k)]
    else:
        attn_adapter_paths = []
        attn_weight_coefs = []

    merged_ffn = 0
    missing_ffn_modules = 0
    merged_attn = 0
    missing_attn_modules = 0

    with torch.no_grad():
        for layer in range(num_layers):
            for e in range(num_experts_cfg):
                ap = adapter_for_expert[e]
                alpha = float(alpha_for_expert[e])
                lora_state_dict = lora_state_dict_for_adapter[ap]
                lora_cfg = lora_cfg_for_adapter[ap]

                for proj in ffn_projs:
                    expert_w_key = f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"
                    if expert_w_key not in state_dict:
                        raise KeyError(
                            f"Expected expert weight key not found: {expert_w_key}"
                        )

                    dense_w_key = f"model.layers.{layer}.mlp.{proj}.weight"
                    lora = _collect_lora_for_target(
                        lora_state_dict=lora_state_dict,
                        adapter_cfg=lora_cfg,
                        target_weight_key=dense_w_key,
                    )
                    if lora is None:
                        missing_ffn_modules += 1
                        continue

                    _apply_lora_delta_inplace(
                        base_weight=state_dict[expert_w_key],
                        lora=lora,
                        alpha_override=alpha,
                        weight_coef=1.0,
                    )
                    merged_ffn += 1

        if merge_attn:
            for layer in range(num_layers):
                for proj in attn_projs:
                    base_w_key = f"model.layers.{layer}.self_attn.{proj}.weight"
                    if base_w_key not in state_dict:
                        raise KeyError(
                            f"Expected attention weight key not found: {base_w_key}"
                        )

                    for ap, coef in zip(attn_adapter_paths, attn_weight_coefs):
                        lora_state_dict = lora_state_dict_for_adapter[ap]
                        lora_cfg = lora_cfg_for_adapter[ap]

                        lora = _collect_lora_for_target(
                            lora_state_dict=lora_state_dict,
                            adapter_cfg=lora_cfg,
                            target_weight_key=base_w_key,
                        )
                        if lora is None:
                            missing_attn_modules += 1
                            continue

                        _apply_lora_delta_inplace(
                            base_weight=state_dict[base_w_key],
                            lora=lora,
                            alpha_override=attn_alpha,
                            weight_coef=coef,
                        )
                        merged_attn += 1

    moe_model.load_state_dict(
        state_dict,
        strict=False,
    )
    tokenizer.save_pretrained(output_dir)
    moe_model.save_pretrained(
        output_dir,
        safe_serialization=bool(d2m_cfg.runtime.safe_serialization),
    )

    manifest = {
        "moe_model_dir": moe_model_dir,
        "output_dir": os.path.abspath(output_dir),
        "mode": mode,
        "num_layers": num_layers,
        "num_experts": num_experts_cfg,
        "targets": OmegaConf.to_container(
            d2m_cfg.targets,
            resolve=True,
        ),
        "merged_ffn_weights": merged_ffn,
        "missing_ffn_modules": missing_ffn_modules,
        "merged_attn_weights": merged_attn,
        "missing_attn_modules": missing_attn_modules,
        "adapter_for_expert": adapter_for_expert,
        "alpha_for_expert": alpha_for_expert,
        "attention_adapters_used": attn_adapter_paths,
        "attention_equal_weight_coefs": attn_weight_coefs,
        "runtime": OmegaConf.to_container(
            d2m_cfg.runtime,
            resolve=True,
        ),
        "hydra_config_resolved": OmegaConf.to_container(
            d2m_cfg,
            resolve=True,
        ),
    }
    with open(
        os.path.join(
            output_dir,
            "merge_manifest.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n[OK] LoRA bake-in finished (FFN + attention optional).")
    print(
        json.dumps(
            {k: manifest[k] for k in manifest if k != "hydra_config_resolved"},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    merge_dense_lora_to_moe()
