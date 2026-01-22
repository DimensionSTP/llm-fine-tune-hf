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

import torch

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)

from peft import PeftModel

import hydra
from omegaconf import DictConfig


def torch_dtype_from_precision(
    precision: object,
) -> torch.dtype:
    if precision in [32, "32"]:
        return torch.float32
    if precision in [16, "16"]:
        return torch.float16
    if precision in ["bf16", "bfloat16"]:
        return torch.bfloat16
    return torch.float32


@hydra.main(
    config_path="../../configs/",
    config_name="sft.yaml",
)
def merge_lora(
    config: DictConfig,
) -> None:
    base_model_name_or_path = config.pretrained_model_name
    adapter_path = config.peft_test.adapter_path
    merged_output_path = os.path.join(
        config.merge_path,
        config.save_detail,
    )

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"Adapter checkpoint directory not found: {adapter_path}"
        )

    os.makedirs(
        merged_output_path,
        exist_ok=True,
    )

    torch_dtype = torch_dtype_from_precision(
        precision=config.precision,
    )

    if config.modality == "text":
        data_encoder = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=base_model_name_or_path,
            revision=str(config.revision),
        )
    else:
        data_encoder = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=base_model_name_or_path,
            revision=str(config.revision),
        )

    if config.modality == "text":
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=None,
            revision=str(config.revision),
        )
    else:
        base_model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=None,
            revision=str(config.revision),
        )
    base_model.eval()

    peft_model = PeftModel.from_pretrained(
        model=base_model,
        model_id=adapter_path,
        is_trainable=False,
    )
    peft_model.eval()

    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    data_encoder.save_pretrained(
        merged_output_path,
    )
    merged_model.save_pretrained(
        merged_output_path,
        safe_serialization=True,
    )

    print(f"[OK] Merged LoRA (and modules_to_save if present) into base model.")
    print(f"[OK] Base model: {base_model_name_or_path}")
    print(f"[OK] Adapter checkpoint: {adapter_path}")
    print(f"[OK] Saved merged model to: {merged_output_path}")


if __name__ == "__main__":
    merge_lora()
