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
    AutoModelForCausalLM,
    AutoTokenizer,
    MixtralConfig,
    MixtralForCausalLM,
)

import hydra
from omegaconf import DictConfig


SUPPORTED_DENSE_MODEL_TYPES = {
    "falcon3",
    "llama",
}


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
    router_param: torch.nn.Module,
    init_type: str,
    gain: float,
) -> None:
    if init_type == "zeros":
        torch.nn.init.zeros_(router_param.weight)
        return
    if init_type == "xavier_uniform":
        torch.nn.init.xavier_uniform_(
            router_param.weight,
            gain=gain,
        )
        return
    if init_type == "kaiming_uniform":
        torch.nn.init.kaiming_uniform_(
            router_param.weight,
            a=0.0,
        )
        return
    raise ValueError(f"Unknown router init_type: {init_type}")


def copy_dense_mlp_to_mixtral_experts(
    moe_model: torch.nn.Module,
    dense_model: torch.nn.Module,
    num_experts: int,
) -> Tuple[int, int]:
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

        gate_proj = dense_mlp.gate_proj.weight.detach().to(torch.float32)
        up_proj = dense_mlp.up_proj.weight.detach().to(torch.float32)
        down_proj = dense_mlp.down_proj.weight.detach().to(torch.float32)

        gate_up_proj = torch.cat(
            [
                gate_proj,
                up_proj,
            ],
            dim=0,
        )

        if moe_mlp.experts.gate_up_proj.shape[0] != num_experts:
            raise ValueError(
                f"Layer {layer_idx}: experts len mismatch: {moe_mlp.experts.gate_up_proj.shape[0]} vs {num_experts}"
            )

        if moe_mlp.experts.gate_up_proj.shape[1:] != gate_up_proj.shape:
            raise ValueError(
                "Layer "
                + f"{layer_idx}: gate_up_proj shape mismatch: "
                + f"{tuple(moe_mlp.experts.gate_up_proj.shape[1:])} vs {tuple(gate_up_proj.shape)}"
            )

        if moe_mlp.experts.down_proj.shape[1:] != down_proj.shape:
            raise ValueError(
                "Layer "
                + f"{layer_idx}: down_proj shape mismatch: "
                + f"{tuple(moe_mlp.experts.down_proj.shape[1:])} vs {tuple(down_proj.shape)}"
            )

        with torch.no_grad():
            for expert_idx in range(num_experts):
                moe_mlp.experts.gate_up_proj[expert_idx].copy_(
                    gate_up_proj.to(
                        device=moe_mlp.experts.gate_up_proj.device,
                        dtype=moe_mlp.experts.gate_up_proj.dtype,
                    )
                )
                moe_mlp.experts.down_proj[expert_idx].copy_(
                    down_proj.to(
                        device=moe_mlp.experts.down_proj.device,
                        dtype=moe_mlp.experts.down_proj.dtype,
                    )
                )
                copied += 3

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

    dense_model = AutoModelForCausalLM.from_pretrained(
        config.pretrained_model_name,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    ).to(device)
    dense_model.eval()

    dense_cfg = dense_model.config
    dense_model_type = str(getattr(dense_cfg, "model_type", ""))
    if dense_model_type not in SUPPORTED_DENSE_MODEL_TYPES:
        raise ValueError(
            "dense_to_moe supports only "
            + f"{sorted(SUPPORTED_DENSE_MODEL_TYPES)}, got '{dense_model_type}'"
        )

    moe_cfg = MixtralConfig.from_dict(dense_cfg.to_dict())
    moe_cfg.model_type = "mixtral"
    moe_cfg.architectures = ["MixtralForCausalLM"]
    moe_cfg.num_local_experts = int(d2m_cfg.moe.num_experts)
    moe_cfg.num_experts_per_tok = int(d2m_cfg.moe.num_experts_per_tok)
    moe_cfg.router_aux_loss_coef = float(d2m_cfg.moe.router_aux_loss_coef)
    moe_cfg.output_router_logits = bool(d2m_cfg.moe.output_router_logits)
    moe_cfg.intermediate_size = int(dense_cfg.intermediate_size)

    moe_model = MixtralForCausalLM(moe_cfg).to(
        device=device,
        dtype=dtype,
    )
    moe_model.eval()

    missing, unexpected = moe_model.load_state_dict(
        dense_model.state_dict(),
        strict=False,
    )
    print(f"[load shared] missing={len(missing)} unexpected={len(unexpected)}")

    copied_tensors, layers_touched = copy_dense_mlp_to_mixtral_experts(
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
        if hasattr(mlp, "gate"):
            init_router_weights(
                router_param=mlp.gate,
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

    print(
        "[OK] Saved Mixtral-style sparse checkpoint to: "
        + f"{d2m_cfg.moe_model_dir} (source_model_type={dense_model_type})"
    )


if __name__ == "__main__":
    dense_to_moe()
