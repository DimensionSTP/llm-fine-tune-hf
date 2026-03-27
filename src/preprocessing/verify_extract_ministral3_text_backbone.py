import dotenv

dotenv.load_dotenv(
    override=True,
)

import json
import os
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["HF_HOME"] = os.environ.get("HF_HOME")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from typing import Dict, List

import hydra
import torch

from omegaconf import DictConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

from src.preprocessing.extract_ministral3_text_backbone import (
    build_extracted_ministral3_state_dict,
    resolve_ministral3_backbone_output_dir,
    torch_dtype_from_str,
)


def _max_abs_diff(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    return float((a - b).abs().max().item())


def _candidate_keys(
    num_layers: int,
) -> List[str]:
    keys = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    for layer in sorted(set([0, num_layers // 2, num_layers - 1])):
        keys.extend(
            [
                f"model.layers.{layer}.self_attn.q_proj.weight",
                f"model.layers.{layer}.self_attn.o_proj.weight",
                f"model.layers.{layer}.mlp.gate_proj.weight",
                f"model.layers.{layer}.mlp.up_proj.weight",
                f"model.layers.{layer}.mlp.down_proj.weight",
            ]
        )
    return keys


@hydra.main(
    config_path="../../configs/",
    config_name="sft.yaml",
)
def verify_extract_ministral3_text_backbone(
    config: DictConfig,
) -> None:
    pretrained_model_name = str(config.pretrained_model_name)
    revision = str(config.revision)
    output_dir = resolve_ministral3_backbone_output_dir(config=config)
    dtype = torch_dtype_from_str(dtype_str=config.precision)

    source_config = AutoConfig.from_pretrained(
        pretrained_model_name,
        revision=revision,
        trust_remote_code=True,
    )
    if str(source_config.model_type) != "mistral3":
        raise ValueError(
            f"Expected top-level Mistral3 VLM config, got model_type={source_config.model_type}"
        )

    source_model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name,
        revision=revision,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None,
    )
    source_model.eval()
    extracted_state_dict = build_extracted_ministral3_state_dict(
        source_state_dict=source_model.state_dict(),
    )

    extracted_model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None,
    )
    extracted_model.eval()
    target_state_dict = extracted_model.state_dict()

    num_layers = int(extracted_model.config.num_hidden_layers)
    keys = _candidate_keys(num_layers=num_layers)
    report: Dict[str, List[Dict[str, float]]] = {"checks": []}

    for key in keys:
        if key not in extracted_state_dict:
            raise KeyError(f"Missing extracted backbone source key: {key}")
        if key not in target_state_dict:
            raise KeyError(f"Missing saved backbone key: {key}")

        diff = _max_abs_diff(
            a=extracted_state_dict[key].detach().cpu(),
            b=target_state_dict[key].detach().cpu(),
        )
        report["checks"].append(
            {
                "key": key,
                "max_abs_diff": diff,
            }
        )
        if diff != 0.0:
            raise RuntimeError(f"Backbone verify failed for key={key}, max_abs_diff={diff}")

    report_path = os.path.join(
        output_dir,
        "verify_extract_ministral3_text_backbone_report.json",
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[OK] Verified extracted Ministral3 text backbone: {output_dir}")


if __name__ == "__main__":
    verify_extract_ministral3_text_backbone()
