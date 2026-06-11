from typing import Dict, Union, Optional
import os
from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf

import torch

from transformers import AutoConfig, BitsAndBytesConfig, PretrainedConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .peft_initialization import (
    is_peft_continue_from_adapter,
    validate_peft_continuation_base_resolution,
)


@dataclass
class ModelLoadPlan:
    pretrained_model_name: str
    pretrained_config: Optional[PretrainedConfig]
    quantization_config: Optional[BitsAndBytesConfig]
    device_map: Union[str, Dict[str, str], None]
    hf_deepspeed_config: Optional[HfDeepSpeedConfig]


class ModelLoadPlanner:
    def __init__(
        self,
        config: DictConfig,
        torch_dtype: Union[torch.dtype, str],
    ) -> None:
        self.config = config
        self.torch_dtype = torch_dtype

    def build(self) -> ModelLoadPlan:
        self.validate()

        pretrained_model_name = self.resolve_pretrained_model_name()
        hf_deepspeed_config = self.build_hf_deepspeed_config()
        pretrained_config = self.build_pretrained_config(
            pretrained_model_name=pretrained_model_name,
        )
        quantization_config = self.build_quantization_config()
        device_map = self.resolve_device_map()

        return ModelLoadPlan(
            pretrained_model_name=pretrained_model_name,
            pretrained_config=pretrained_config,
            quantization_config=quantization_config,
            device_map=device_map,
            hf_deepspeed_config=hf_deepspeed_config,
        )

    def validate(self) -> None:
        self.validate_deepspeed_stage()
        self.validate_deepspeed_zero3_init()
        self.validate_quantized_training()
        self.validate_fsdp_training()

    def resolve_pretrained_model_name(self) -> str:
        pretrained_model_name = self.config.pretrained_model_name
        if self.is_inference() or not self.config.is_preprocessed:
            return pretrained_model_name

        if is_peft_continue_from_adapter(config=self.config):
            validate_peft_continuation_base_resolution(config=self.config)
            return pretrained_model_name

        merged_model_path = os.path.join(
            self.config.merged_model_path,
            self.config.pretrained_model_name,
        )
        if os.path.exists(merged_model_path):
            return merged_model_path
        return pretrained_model_name

    def build_hf_deepspeed_config(self) -> Optional[HfDeepSpeedConfig]:
        if not self.should_activate_deepspeed_zero3_init():
            return None

        deepspeed_config = OmegaConf.to_container(
            self.config.deepspeed,
            resolve=True,
        )
        return HfDeepSpeedConfig(deepspeed_config)

    def build_pretrained_config(
        self,
        pretrained_model_name: str,
    ) -> Optional[PretrainedConfig]:
        if self.config.is_quantized:
            return None

        pretrained_config = AutoConfig.from_pretrained(
            pretrained_model_name,
            revision=self.config.revision,
            trust_remote_code=True,
        )
        if getattr(pretrained_config, "quantization_config", None) is None:
            pretrained_config_dict = pretrained_config.to_dict()
            pretrained_config_dict.pop(
                "quantization_config",
                None,
            )
            pretrained_config = pretrained_config.__class__.from_dict(
                pretrained_config_dict,
            )
        pretrained_config.output_hidden_states = False
        return pretrained_config

    def build_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        if self.is_inference() or not self.config.is_quantized:
            return None

        quantization_config_dict = OmegaConf.to_container(
            self.config.quantization_config,
            resolve=True,
        )
        if self.should_use_qlora_deepspeed_zero3():
            quantization_config_dict["bnb_4bit_quant_storage"] = (
                self.resolve_torch_dtype(
                    dtype_name=self.config.model_loading.qlora.quant_storage_dtype,
                )
            )

        return BitsAndBytesConfig(**quantization_config_dict)

    def resolve_device_map(self) -> Union[str, Dict[str, str], None]:
        if self.config.mode == "test_large":
            return self.normalize_device_map(
                device_map=self.config.model_loading.inference.test_large_device_map,
            )

        if self.is_inference():
            return self.normalize_device_map(
                device_map=self.config.model_loading.inference.device_map,
            )

        train_device_map = self.config.model_loading.train.device_map
        if train_device_map is not None:
            return self.normalize_device_map(device_map=train_device_map)

        if not self.config.is_quantized:
            return None

        if self.should_use_qlora_deepspeed_zero3():
            return None

        if self.config.model_loading.train.quantized_local_rank_device_map:
            return {"": f"cuda:{self.get_local_rank()}"}

        return None

    def normalize_device_map(
        self,
        device_map: object,
    ) -> Union[str, Dict[str, str], None]:
        if OmegaConf.is_config(device_map):
            return OmegaConf.to_container(
                device_map,
                resolve=True,
            )
        if device_map is None:
            return None
        return str(device_map)

    def should_activate_deepspeed_zero3_init(self) -> bool:
        if self.is_inference():
            return False

        if self.config.strategy != "deepspeed":
            return False

        if self.get_deepspeed_stage() != 3:
            return False

        zero3_init = self.config.model_loading.deepspeed.zero3_init
        if zero3_init == "disabled":
            return False

        if self.config.is_quantized:
            return self.should_use_qlora_deepspeed_zero3()

        return zero3_init in ["auto", "enabled"]

    def should_use_qlora_deepspeed_zero3(self) -> bool:
        if not self.config.is_quantized:
            return False

        return self.config.model_loading.qlora.deepspeed_zero3_enabled

    def validate_deepspeed_stage(self) -> None:
        if self.config.strategy != "deepspeed":
            return

        deepspeed_stage = self.get_deepspeed_stage()
        if deepspeed_stage in [0, 1, 2, 3]:
            return

        raise ValueError(
            "deepspeed.zero_optimization.stage must be one of 0, 1, 2, or 3."
        )

    def validate_deepspeed_zero3_init(self) -> None:
        zero3_init = self.config.model_loading.deepspeed.zero3_init
        if zero3_init not in ["auto", "enabled", "disabled"]:
            raise ValueError(
                "model_loading.deepspeed.zero3_init must be auto, enabled, or disabled."
            )

        if zero3_init != "enabled":
            return

        if self.config.strategy == "deepspeed" and self.get_deepspeed_stage() == 3:
            return

        raise ValueError(
            "model_loading.deepspeed.zero3_init=enabled requires "
            "strategy=deepspeed and deepspeed.zero_optimization.stage=3."
        )

    def validate_quantized_training(self) -> None:
        if self.is_inference() or not self.config.is_quantized:
            return

        if not self.should_use_qlora_deepspeed_zero3():
            return

        if not self.config.is_peft:
            raise ValueError(
                "model_loading.qlora.deepspeed_zero3_enabled=true requires is_peft=true."
            )

        if not OmegaConf.select(
            self.config,
            "quantization_config.load_in_4bit",
            default=False,
        ):
            raise ValueError(
                "model_loading.qlora.deepspeed_zero3_enabled=true requires "
                "quantization_config.load_in_4bit=true."
            )

        if self.config.strategy != "deepspeed" or self.get_deepspeed_stage() != 3:
            raise ValueError(
                "model_loading.qlora.deepspeed_zero3_enabled=true requires "
                "strategy=deepspeed and deepspeed.zero_optimization.stage=3."
            )

    def validate_fsdp_training(self) -> None:
        if self.config.strategy != "fsdp":
            return

        if not self.config.model_loading.fsdp.require_training_arguments:
            return

        if OmegaConf.select(self.config, "training_arguments.fsdp") is not None:
            return

        raise ValueError(
            "strategy=fsdp requires explicit training_arguments.fsdp config in this repo."
        )

    def get_deepspeed_stage(self) -> Optional[int]:
        if self.config.strategy != "deepspeed":
            return None

        deepspeed_stage = OmegaConf.select(
            self.config,
            "deepspeed.zero_optimization.stage",
            default=0,
        )
        return int(deepspeed_stage)

    def get_local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK") or 0)

    def is_inference(self) -> bool:
        return self.config.mode in ["test", "test_large"]

    def resolve_torch_dtype(
        self,
        dtype_name: Union[torch.dtype, str],
    ) -> Union[torch.dtype, str]:
        if isinstance(dtype_name, torch.dtype):
            return dtype_name

        if dtype_name == "bfloat16":
            return torch.bfloat16
        if dtype_name == "float16":
            return torch.float16
        if dtype_name == "float32":
            return torch.float32
        if dtype_name == "auto":
            return "auto"

        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
