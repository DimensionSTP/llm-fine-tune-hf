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

from transformers import AutoModelForCausalLM

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


def _get_expert_proj_tensor(
    state_dict: Dict[str, torch.Tensor],
    layer: int,
    expert: int,
    proj: str,
) -> torch.Tensor:
    if proj in [
        "gate_proj",
        "up_proj",
    ]:
        gate_up_key = f"model.layers.{layer}.mlp.experts.gate_up_proj"
        if gate_up_key not in state_dict:
            raise KeyError(f"Missing expected Mixtral key: {gate_up_key}")
        gate_up = state_dict[gate_up_key][expert]
        intermediate_size = gate_up.shape[0] // 2
        if proj == "gate_proj":
            return gate_up[:intermediate_size]
        return gate_up[intermediate_size:]

    if proj == "down_proj":
        down_key = f"model.layers.{layer}.mlp.experts.down_proj"
        if down_key not in state_dict:
            raise KeyError(f"Missing expected Mixtral key: {down_key}")
        return state_dict[down_key][expert]

    raise ValueError(f"Unknown proj={proj}")


@hydra.main(
    config_path="../../configs/",
    config_name="sft.yaml",
)
def verify_dense_to_moe_mixtral(
    config: DictConfig,
) -> None:
    d2m_cfg = config.dense_to_moe

    moe_dir = str(d2m_cfg.moe_model_dir)
    dense_id = str(config.pretrained_model_name)

    dtype = torch_dtype_from_str(dtype_str=d2m_cfg.runtime.dtype)
    device = torch.device(str(d2m_cfg.runtime.device))
    trust_remote_code = bool(d2m_cfg.runtime.trust_remote_code)

    dense_model = AutoModelForCausalLM.from_pretrained(
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
    if model_type != "mixtral":
        raise RuntimeError(f"Expected model_type 'mixtral', got '{model_type}'")

    for k in [
        "num_local_experts",
        "num_experts_per_tok",
        "intermediate_size",
    ]:
        if k not in cfg_dict:
            raise RuntimeError(f"Missing MoE field in config: {k}")
    print(
        "[config] num_local_experts="
        + f"{cfg_dict['num_local_experts']}, top_k={cfg_dict['num_experts_per_tok']}, intermediate_size={cfg_dict['intermediate_size']}"
    )

    num_layers = _infer_num_layers(model=moe_model)
    num_experts = int(cfg_dict["num_local_experts"])

    for layer in range(num_layers):
        gate_key = f"model.layers.{layer}.mlp.gate.weight"
        gate_up_key = f"model.layers.{layer}.mlp.experts.gate_up_proj"
        down_key = f"model.layers.{layer}.mlp.experts.down_proj"
        if gate_key not in moe_state_dict:
            raise RuntimeError(f"Layer {layer}: missing gate key {gate_key}")
        if gate_up_key not in moe_state_dict:
            raise RuntimeError(f"Layer {layer}: missing expert key {gate_up_key}")
        if down_key not in moe_state_dict:
            raise RuntimeError(f"Layer {layer}: missing expert key {down_key}")

        if moe_state_dict[gate_up_key].shape[0] != num_experts:
            raise RuntimeError(
                f"Layer {layer}: gate_up_proj experts {moe_state_dict[gate_up_key].shape[0]} != {num_experts}"
            )
        if moe_state_dict[down_key].shape[0] != num_experts:
            raise RuntimeError(
                f"Layer {layer}: down_proj experts {moe_state_dict[down_key].shape[0]} != {num_experts}"
            )

    verify_cfg = getattr(d2m_cfg, "verify_merge", {})
    verify_layers = list(getattr(verify_cfg, "layers", [0, num_layers // 2, num_layers - 1]))
    verify_experts = list(getattr(verify_cfg, "experts", [0, num_experts - 1]))
    verify_projs = list(getattr(verify_cfg, "projs", ["up_proj", "gate_proj", "down_proj"]))

    verify_layers = [layer for layer in verify_layers if 0 <= int(layer) < num_layers]
    verify_experts = [expert for expert in verify_experts if 0 <= int(expert) < num_experts]
    tol = float(getattr(verify_cfg, "weight_tol", 0.0))

    report = {"mlp_copy_checks": []}

    for layer in verify_layers:
        for expert in verify_experts:
            for proj in verify_projs:
                dense_key = f"model.layers.{int(layer)}.mlp.{proj}.weight"
                if dense_key not in dense_sd:
                    raise KeyError(f"Missing dense key: {dense_key}")

                dense_tensor = dense_sd[dense_key].detach().cpu()
                expert_tensor = _get_expert_proj_tensor(
                    state_dict=moe_state_dict,
                    layer=int(layer),
                    expert=int(expert),
                    proj=proj,
                ).detach().cpu()

                mad = _max_abs_diff(
                    a=dense_tensor,
                    b=expert_tensor,
                )
                meanad = _mean_abs_diff(
                    a=dense_tensor,
                    b=expert_tensor,
                )
                cos = _cosine_sim(
                    a=dense_tensor,
                    b=expert_tensor,
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
                        f"[MLP copy mismatch] layer={layer} expert={expert} proj={proj} max_abs_diff={mad} (tol={tol})"
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
                if init_type == "zeros" and float(w.abs().max().item()) != 0.0:
                    raise RuntimeError(
                        f"[router init mismatch] layer={layer} key={gate_key} is not zero-initialized"
                    )
                break

        if not found:
            raise RuntimeError(f"Layer {layer}: no gate/router key found")

    report["gate_stats"] = gate_stats

    out_path = os.path.join(moe_dir, "verify_dense_to_moe_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[OK] Dense-to-MoE verification report saved to: {out_path}")


if __name__ == "__main__":
    verify_dense_to_moe_mixtral()
