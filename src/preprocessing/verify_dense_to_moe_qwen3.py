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

from typing import Dict, List
import json

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM

import hydra
from omegaconf import DictConfig


def torch_dtype_from_str(dtype_str: str) -> torch.dtype:
    dtype_str = str(dtype_str).lower()
    if dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_str in ("fp16", "float16"):
        return torch.float16
    if dtype_str in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}")


def _infer_num_layers(model: torch.nn.Module) -> int:
    return len(model.model.layers)


def _find_dense_mlp_weight_key(
    layer: int,
    proj: str,
) -> str:
    return f"model.layers.{layer}.mlp.{proj}.weight"


def _find_expert_weight_key(
    state_dict: Dict[str, torch.Tensor],
    layer: int,
    expert: int,
    proj: str,
) -> str:
    k = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
    if k not in state_dict:
        raise KeyError(f"Missing expected expert weight key: {k}")
    return k


def _max_abs_diff(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    return float((a - b).abs().max().item())


def _mean_abs_diff(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    return float((a - b).abs().mean().item())


def _cosine_sim(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return float(
        torch.dot(
            a,
            b,
        )
        / (a.norm() * b.norm() + eps)
    )


def _gate_key_candidates(layer: int) -> List[str]:
    return [
        f"model.layers.{layer}.mlp.gate.weight",
        f"model.layers.{layer}.mlp.router.weight",
    ]


@hydra.main(
    config_path="../../configs/",
    config_name="sft.yaml",
)
def verify_dense_to_moe(
    config: DictConfig,
) -> None:
    d2m_cfg = config.dense_to_moe

    moe_dir = str(d2m_cfg.moe_model_dir)
    dense_id = str(config.pretrained_model_name)

    dtype = torch_dtype_from_str(dtype_str=d2m_cfg.runtime.dtype)
    device = torch.device(str(d2m_cfg.runtime.device))
    trust_remote_code = bool(d2m_cfg.runtime.trust_remote_code)

    dense_model = Qwen3ForCausalLM.from_pretrained(
        dense_id,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    ).to(device)
    dense_model.eval()
    dense_sd = dense_model.state_dict()

    moe_model = AutoModelForCausalLM.from_pretrained(
        moe_dir,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    ).to(device)
    moe_model.eval()
    moe_state_dict = moe_model.state_dict()

    cfg_dict = moe_model.config.to_dict()
    model_type = cfg_dict.get("model_type", None)
    print(f"[config] model_type={model_type}")
    if model_type != "qwen3_moe":
        raise RuntimeError(f"Expected model_type 'qwen3_moe', got '{model_type}'")

    for k in ["num_experts", "num_experts_per_tok", "moe_intermediate_size"]:
        if k not in cfg_dict:
            raise RuntimeError(f"Missing MoE field in config: {k}")
    print(
        f"[config] num_experts={cfg_dict['num_experts']}, top_k={cfg_dict['num_experts_per_tok']}, moe_intermediate_size={cfg_dict['moe_intermediate_size']}"
    )

    num_layers = _infer_num_layers(model=moe_model)
    num_experts = int(cfg_dict["num_experts"])
    touched_layers = 0

    for l in range(num_layers):
        mlp = moe_model.model.layers[l].mlp
        if hasattr(mlp, "experts"):
            if len(mlp.experts) != num_experts:
                raise RuntimeError(
                    f"Layer {l}: experts len {len(mlp.experts)} != config num_experts {num_experts}"
                )
            touched_layers += 1
        else:
            raise RuntimeError(
                f"Layer {l}: no 'experts' attribute found in mlp (expected MoE every layer with decoder_sparse_step=1)."
            )

        if not (hasattr(mlp, "gate") or hasattr(mlp, "router")):
            raise RuntimeError(f"Layer {l}: no gate/router attribute found in mlp.")

    print(
        f"[structure] layers={num_layers}, moe_layers_detected={touched_layers}, experts={num_experts}"
    )

    verify_layers = list(
        getattr(d2m_cfg, "verify", {}).get(
            "layers", [0, num_layers // 2, num_layers - 1]
        )
    )
    verify_experts = list(
        getattr(d2m_cfg, "verify", {}).get("experts", [0, num_experts - 1])
    )
    verify_projs = list(
        getattr(d2m_cfg, "verify", {}).get(
            "projs", ["up_proj", "gate_proj", "down_proj"]
        )
    )

    verify_layers = [l for l in verify_layers if 0 <= int(l) < num_layers]
    verify_experts = [e for e in verify_experts if 0 <= int(e) < num_experts]

    tol = float(getattr(d2m_cfg, "verify", {}).get("weight_tol", 0.0))

    report = {"mlp_copy_checks": []}

    for layer in verify_layers:
        for expert in verify_experts:
            for proj in verify_projs:
                dense_k = _find_dense_mlp_weight_key(
                    layer=layer,
                    proj=proj,
                )
                expert_k = _find_expert_weight_key(
                    state_dict=moe_state_dict,
                    layer=layer,
                    expert=expert,
                    proj=proj,
                )

                a = dense_sd[dense_k].detach().to("cpu")
                b = moe_state_dict[expert_k].detach().to("cpu")

                mad = _max_abs_diff(
                    a=a,
                    b=b,
                )
                meanad = _mean_abs_diff(
                    a=a,
                    b=b,
                )
                cos = _cosine_sim(
                    a=a,
                    b=b,
                )

                ok = mad <= tol if tol > 0 else mad == 0.0
                report["mlp_copy_checks"].append(
                    {
                        "layer": int(layer),
                        "expert": int(expert),
                        "proj": proj,
                        "max_abs_diff": mad,
                        "mean_abs_diff": meanad,
                        "cosine": cos,
                        "pass": bool(ok),
                    }
                )

                if not ok:
                    raise RuntimeError(
                        f"[MLP copy mismatch] layer={layer} expert={expert} proj={proj} "
                        f"max_abs_diff={mad} (tol={tol})"
                    )

    print(
        f"[verify] MLP copy checks passed for layers={verify_layers}, experts={verify_experts}, projs={verify_projs}"
    )

    init_type = str(d2m_cfg.router.init_type)
    gate_stats = []
    for layer in verify_layers:
        found = False
        for gate_key in _gate_key_candidates(layer=layer):
            if gate_key in moe_state_dict:
                w = moe_state_dict[gate_key].detach().float().cpu()
                gate_stats.append(
                    {
                        "layer": int(layer),
                        "gate_key": gate_key,
                        "mean": float(w.mean().item()),
                        "std": float(w.std().item()),
                        "max_abs": float(w.abs().max().item()),
                    }
                )
                found = True
                if init_type == "zeros":
                    if float(w.abs().max().item()) != 0.0:
                        raise RuntimeError(
                            f"[router init mismatch] expected zeros but {gate_key} max_abs={float(w.abs().max().item())}"
                        )
                break
        if not found:
            raise RuntimeError(
                f"Could not find gate/router weight key for layer {l} in state_dict."
            )

    print(
        f"[verify] Router init looks consistent with init_type={init_type} (sampled layers={verify_layers})"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        moe_dir,
        trust_remote_code=trust_remote_code,
    )
    text = "Hello, this is a router verification test."
    inputs = tokenizer(
        text,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = moe_model(
            **inputs,
            output_router_logits=True,
            use_cache=False,
        )

    if not hasattr(out, "router_logits") or out.router_logits is None:
        raise RuntimeError(
            "Model forward did not return router_logits even with output_router_logits=True."
        )
    rl = out.router_logits
    if not isinstance(rl, (tuple, list)) or len(rl) != num_layers:
        raise RuntimeError(
            f"router_logits expected length {num_layers}, got {type(rl)} len={len(rl) if isinstance(rl,(tuple,list)) else 'N/A'}"
        )

    sample_rl = rl[verify_layers[0]]
    if sample_rl.shape[-1] != num_experts:
        raise RuntimeError(
            f"router_logits last dim expected num_experts={num_experts}, got {sample_rl.shape}"
        )

    print(
        f"[verify] Forward router_logits OK: len={len(rl)}, sample_shape={tuple(sample_rl.shape)}"
    )

    out_path = os.path.join(moe_dir, "verify_dense_to_moe_report.json")
    report["config"] = {
        "model_type": model_type,
        "num_layers": num_layers,
        "num_experts": num_experts,
    }
    report["router_gate_stats_sample"] = gate_stats
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[OK] Verification report saved to: {out_path}")


if __name__ == "__main__":
    verify_dense_to_moe()
