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

from typing import Dict

import hydra
import torch

from omegaconf import DictConfig
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import Ministral3ForCausalLM


def resolve_ministral3_backbone_output_dir(
    config: DictConfig,
) -> str:
    return os.path.join(
        config.connected_dir,
        "extract_ministral3_text_backbone",
        config.model_type,
    )


def torch_dtype_from_str(dtype_str: str) -> torch.dtype:
    dtype_str = str(dtype_str).lower()
    if dtype_str in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_str in ("fp16", "float16"):
        return torch.float16
    if dtype_str in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}")


def build_extracted_ministral3_state_dict(
    source_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    extracted_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in source_state_dict.items():
        if key.startswith("model.language_model."):
            extracted_key = "model." + key[len("model.language_model.") :]
            extracted_state_dict[extracted_key] = value
        elif key.startswith("lm_head."):
            extracted_state_dict[key] = value
    return extracted_state_dict


def load_source_text_tokenizer(
    pretrained_model_name: str,
    revision: str,
) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            revision=revision,
        )
        return tokenizer
    except Exception:
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name,
            revision=revision,
        )
        return processor.tokenizer


@hydra.main(
    config_path="../../configs/",
    config_name="sft.yaml",
)
def extract_ministral3_text_backbone(
    config: DictConfig,
) -> None:
    output_dir = resolve_ministral3_backbone_output_dir(config=config)
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    config_path = os.path.join(
        output_dir,
        "config.json",
    )
    if os.path.exists(config_path):
        print(f"[skip] extracted backbone already exists: {output_dir}")
        return

    pretrained_model_name = str(config.pretrained_model_name)
    revision = str(config.revision)
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

    text_config = source_config.text_config
    if str(text_config.model_type) != "ministral3":
        raise ValueError(
            f"Expected text backbone model_type=ministral3, got {text_config.model_type}"
        )

    source_model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name,
        revision=revision,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None,
    )
    source_model.eval()

    target_config = Ministral3Config.from_dict(text_config.to_dict())
    target_config.architectures = ["Ministral3ForCausalLM"]
    target_model = Ministral3ForCausalLM(target_config)
    target_model.eval()

    extracted_state_dict = build_extracted_ministral3_state_dict(
        source_state_dict=source_model.state_dict(),
    )
    missing, unexpected = target_model.load_state_dict(
        extracted_state_dict,
        strict=False,
    )
    if len(unexpected) > 0:
        raise RuntimeError(f"Unexpected target keys while loading extracted backbone: {unexpected}")
    if len(missing) > 0:
        raise RuntimeError(f"Missing target keys while loading extracted backbone: {missing}")

    tokenizer = load_source_text_tokenizer(
        pretrained_model_name=pretrained_model_name,
        revision=revision,
    )
    tokenizer.save_pretrained(output_dir)
    target_model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )

    manifest = {
        "source_model": pretrained_model_name,
        "source_model_type": str(source_config.model_type),
        "text_backbone_model_type": str(text_config.model_type),
        "output_dir": output_dir,
        "num_extracted_tensors": len(extracted_state_dict),
    }
    manifest_path = os.path.join(
        output_dir,
        "extract_ministral3_text_backbone_manifest.json",
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[OK] Extracted Ministral3 text backbone to: {output_dir}")


if __name__ == "__main__":
    extract_ministral3_text_backbone()
