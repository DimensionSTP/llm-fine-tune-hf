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

from typing import Tuple

import torch

from transformers import (
    AutoTokenizer,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

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


def init_router_weights(
    router_linear: torch.nn.Linear,
    init_type: str,
    gain: float,
) -> None:
    if init_type == "zeros":
        torch.nn.init.zeros_(router_linear.weight)
        return
    if init_type == "xavier_uniform":
        torch.nn.init.xavier_uniform_(
            router_linear.weight,
            gain=gain,
        )
        return
    if init_type == "kaiming_uniform":
        torch.nn.init.kaiming_uniform_(
            router_linear.weight,
            a=0.0,
        )
        return
    raise ValueError(f"Unknown router init_type: {init_type}")


def copy_dense_mlp_to_all_experts(
    moe_model: torch.nn.Module,
    dense_model: torch.nn.Module,
    num_experts: int,
) -> Tuple[int, int]:
    """
    Copy dense layer MLP weights to each expert MLP weights.
    Returns: (num_copied_tensors, num_layers_touched)
    """
    copied = 0
    layers = 0

    moe_layers = moe_model.model.layers
    dense_layers = dense_model.model.layers

    if len(moe_layers) != len(dense_layers):
        raise ValueError(
            f"num_hidden_layers mismatch: moe={len(moe_layers)} dense={len(dense_layers)}"
        )

    for layer_idx in range(len(moe_layers)):
        moe_mlp = moe_layers[layer_idx].mlp
        dense_mlp = dense_layers[layer_idx].mlp

        if not hasattr(moe_mlp, "experts"):
            continue

        experts = moe_mlp.experts
        if len(experts) != num_experts:
            raise ValueError(
                f"Layer {layer_idx}: experts len mismatch: {len(experts)} vs {num_experts}"
            )

        dense_sd = dense_mlp.state_dict()
        for expert_idx in range(num_experts):
            experts[expert_idx].load_state_dict(
                dense_sd,
                strict=True,
            )
            copied += len(dense_sd)

        layers += 1

    return copied, layers


@hydra.main(
    config_path="../../configs/",
    config_name="sft.yaml",
)
def dense_to_moe(
    config: DictConfig,
) -> None:
    d2m_cfg = config.dense_to_moe

    os.makedirs(
        d2m_cfg.moe_model_dir,
        exist_ok=True,
    )

    dtype = torch_dtype_from_str(dtype_str=d2m_cfg.runtime.dtype)
    device = torch.device(str(d2m_cfg.runtime.device))
    trust_remote_code = bool(d2m_cfg.runtime.trust_remote_code)

    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_name,
        trust_remote_code=trust_remote_code,
    )

    dense_model = Qwen3ForCausalLM.from_pretrained(
        config.pretrained_model_name,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    ).to(device)
    dense_model.eval()

    dense_cfg = dense_model.config
    moe_cfg = Qwen3MoeConfig.from_dict(dense_cfg.to_dict())

    moe_cfg.model_type = "qwen3_moe"
    moe_cfg.num_experts = int(d2m_cfg.moe.num_experts)
    moe_cfg.num_experts_per_tok = int(d2m_cfg.moe.num_experts_per_tok)
    moe_cfg.norm_topk_prob = bool(d2m_cfg.moe.norm_topk_prob)
    moe_cfg.router_aux_loss_coef = float(d2m_cfg.moe.router_aux_loss_coef)
    moe_cfg.output_router_logits = bool(d2m_cfg.moe.output_router_logits)

    moe_cfg.moe_intermediate_size = int(dense_cfg.intermediate_size)

    moe_cfg.decoder_sparse_step = 1
    moe_cfg.mlp_only_layers = []

    moe_model = Qwen3MoeForCausalLM(moe_cfg).to(
        device=device,
        dtype=dtype,
    )
    moe_model.eval()

    missing, unexpected = moe_model.load_state_dict(
        dense_model.state_dict(),
        strict=False,
    )
    print(f"[load shared] missing={len(missing)} unexpected={len(unexpected)}")

    copied_tensors, layers_touched = copy_dense_mlp_to_all_experts(
        moe_model=moe_model,
        dense_model=dense_model,
        num_experts=int(d2m_cfg.moe.num_experts),
    )
    print(
        f"[copy dense->experts] layers_touched={layers_touched}, copied_tensors={copied_tensors}"
    )

    init_type = str(d2m_cfg.router.init_type)
    gain = float(d2m_cfg.router.gain)

    routers_inited = 0
    for layer in moe_model.model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "gate") and isinstance(mlp.gate, torch.nn.Linear):
            init_router_weights(
                mlp.gate,
                init_type=init_type,
                gain=gain,
            )
            routers_inited += 1
    print(f"[router init] inited={routers_inited}")

    tokenizer.save_pretrained(d2m_cfg.moe_model_dir)
    moe_model.save_pretrained(
        d2m_cfg.moe_model_dir,
        safe_serialization=bool(d2m_cfg.runtime.safe_serialization),
    )

    print(f"[OK] Saved official Qwen3MoE checkpoint to: {d2m_cfg.moe_model_dir}")


if __name__ == "__main__":
    dense_to_moe()
