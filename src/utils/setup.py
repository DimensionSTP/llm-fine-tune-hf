from typing import Dict, Union, Optional, Any
import os
import re

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import importlib

datasets = importlib.import_module("datasets")
HFDataset = datasets.Dataset

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    PreTrainedTokenizer,
    ProcessorMixin,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PreTrainedModel,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model

from ..datasets import *
from src.utils.rewards import RewardManager


class SetUp:
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.data_type = self.config.data_type
        self.revision = self.config.revision
        self.num_cpus = os.cpu_count()
        self.num_fit_workers = min(
            self.num_cpus,
            (config.devices * config.workers_ratio),
        )
        self.num_workers = (
            self.num_cpus if config.use_all_workers else self.num_fit_workers
        )

        if config.precision in [32, "32"]:
            self.torch_dtype = torch.float32
        elif config.precision in [16, "16"]:
            self.torch_dtype = torch.float16
        elif config.precision == "bf16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = "auto"

    def get_train_dataset(self) -> Dataset:
        train_dataset: Dataset = instantiate(
            self.config.dataset[self.data_type],
            split=self.config.split.train,
        )
        return train_dataset

    def get_val_dataset(self) -> Dataset:
        val_dataset: Dataset = instantiate(
            self.config.dataset[self.data_type],
            split=self.config.split.val,
        )
        return val_dataset

    def get_dataset(self) -> Dict[str, HFDataset]:
        dataset: Union[
            DPOStructuralDataset,
            DPOConversationalDataset,
            GRPOStructuralDataset,
            GRPOConversationalDataset,
        ] = instantiate(
            self.config.dataset[self.data_type],
        )
        return dataset()

    def get_test_dataset(self) -> Dataset:
        test_dataset: Dataset = instantiate(
            self.config.test_dataset[self.data_type],
        )
        return test_dataset

    def get_model(self) -> PreTrainedModel:
        is_inference = getattr(self.config, "mode", "train") in ["test", "test_large"]
        pretrained_model_name = self.config.pretrained_model_name
        if not is_inference and self.config.is_preprocessed:
            merged_model_path = os.path.join(
                self.config.merged_model_path,
                self.config.pretrained_model_name,
            )
            if os.path.exists(merged_model_path):
                pretrained_model_name = merged_model_path

        quantization_config = None
        device_map = None
        if not is_inference and self.config.is_quantized:
            quantization_config_dict = OmegaConf.to_container(
                self.config.quantization_config,
                resolve=True,
            )
            quantization_config = BitsAndBytesConfig(**quantization_config_dict)
            if device_map is None:
                device_map = {"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))}

        if getattr(self.config, "mode", "train") == "test_large":
            device_map = "auto"

        if self.config.modality == "text":
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name,
                output_hidden_states=False,
                torch_dtype=self.torch_dtype,
                attn_implementation=self.config.attn_implementation,
                quantization_config=quantization_config,
                device_map=device_map,
                revision=self.revision,
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name,
                output_hidden_states=False,
                torch_dtype=self.torch_dtype,
                attn_implementation=self.config.attn_implementation,
                quantization_config=quantization_config,
                device_map=device_map,
                revision=self.revision,
            )

        if is_inference:
            model.eval()
            if self.config.is_peft:
                model = PeftModel.from_pretrained(
                    model=model,
                    model_id=self.config.peft_test.adapter_path,
                    adapter_name=self.config.peft_test.adapter_name,
                )

            return model

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=self.config.gradient_checkpointing_kwargs,
            )

        if self.config.dense_to_moe.router_only_train:
            if self.config.is_peft or self.config.is_quantized:
                raise ValueError("router_only_train is not supported with PEFT.")

            if self.config.dense_to_moe.router_with_lora:
                raise ValueError(
                    "router_with_lora is not supported with router_only_train."
                )

            router_regex = self.config.dense_to_moe.router_regex

            for param in model.parameters():
                param.requires_grad = False

            router_pattern = re.compile(router_regex)

            unfrozen = 0
            total = 0
            for name, p in model.named_parameters():
                total += 1
                if router_pattern.search(name):
                    p.requires_grad = True
                    unfrozen += 1

            if unfrozen == 0:
                raise RuntimeError(
                    "No router parameters were unfrozen. "
                    f"Check router_regex='{router_regex}' against model.named_parameters()."
                )

            print(f"[router-only] Unfroze {unfrozen} / {total} parameters (tensors).")

        if self.config.is_quantized and self.config.quantization_config.get(
            "load_in_4bit",
            False,
        ):
            model = prepare_model_for_kbit_training(model)

        if self.config.is_peft:
            peft_config_dict = OmegaConf.to_container(
                self.config.peft_config,
                resolve=True,
            )

            if self.config.dense_to_moe.router_with_lora:
                peft_config_dict["modules_to_save"] = ["gate"]

            peft_config = LoraConfig(**peft_config_dict)
            model = get_peft_model(
                model=model,
                peft_config=peft_config,
            )

        if self.config.dense_to_moe.router_with_lora:
            router_regex = self.config.dense_to_moe.router_lora_regex

            for param in model.parameters():
                param.requires_grad = False

            router_pattern = re.compile(router_regex)

            unfrozen_router = 0
            unfrozen_lora = 0
            total = 0

            for name, p in model.named_parameters():
                total += 1

                if router_pattern.search(name):
                    p.requires_grad = True
                    unfrozen_router += 1
                    continue

                if "lora_" in name:
                    p.requires_grad = True
                    unfrozen_lora += 1
                    continue

            if unfrozen_router == 0:
                raise RuntimeError(
                    "No router parameters were unfrozen. "
                    f"Check router_regex='{router_regex}' against model.named_parameters()."
                )

            if unfrozen_lora == 0:
                raise RuntimeError(
                    "No LoRA parameters were unfrozen. "
                    "Check that get_peft_model injected LoRA modules correctly."
                )

            print(
                f"[router-with-lora] Unfroze router={unfrozen_router}, lora={unfrozen_lora} / {total} parameters (tensors). "
            )

        return model

    def get_data_encoder(self) -> Union[PreTrainedTokenizer, ProcessorMixin]:
        if self.config.is_preprocessed:
            data_encoder_path = self.config.custom_data_encoder_path
        else:
            data_encoder_path = self.config.pretrained_model_name

        if self.config.modality == "text":
            data_encoder = AutoTokenizer.from_pretrained(
                data_encoder_path,
                use_fast=True,
                revision=self.revision,
            )

            if data_encoder.chat_template is None:
                reference_data_encoder = AutoTokenizer.from_pretrained(
                    self.config.reference_data_encoder_name
                )
                data_encoder.chat_template = reference_data_encoder.chat_template

            if data_encoder.pad_token_id is None:
                data_encoder.pad_token_id = data_encoder.eos_token_id
            if self.config.left_padding:
                data_encoder.padding_side = "left"
            else:
                data_encoder.padding_side = "right"
        else:
            data_encoder_kwargs = {}
            if self.config.max_pixels is not None and self.config.max_pixels > 0:
                data_encoder_kwargs["max_pixels"] = self.config.max_pixels
            data_encoder = AutoProcessor.from_pretrained(
                data_encoder_path,
                revision=self.revision,
                **data_encoder_kwargs,
            )

            if data_encoder.tokenizer.chat_template is None:
                reference_data_encoder = AutoTokenizer.from_pretrained(
                    self.config.reference_data_encoder_name
                )
                data_encoder.tokenizer.chat_template = (
                    reference_data_encoder.chat_template
                )

            if data_encoder.tokenizer.pad_token_id is None:
                data_encoder.tokenizer.pad_token_id = (
                    data_encoder.tokenizer.eos_token_id
                )
            if self.config.left_padding:
                data_encoder.tokenizer.padding_side = "left"
            else:
                data_encoder.tokenizer.padding_side = "right"

        return data_encoder

    def get_training_arguments(self) -> TrainingArguments:
        training_arguments: TrainingArguments = instantiate(
            self.config.training_arguments,
            dataloader_num_workers=self.num_workers,
        )
        return training_arguments

    def get_ds_config(self) -> Optional[Dict[str, Any]]:
        if self.config.strategy == "deepspeed":
            ds_config = OmegaConf.to_container(
                self.config.deepspeed,
                resolve=True,
            )
            return ds_config
        return None

    def get_reward_manager(self) -> RewardManager:
        reward_manager: RewardManager = instantiate(
            self.config.reward_manager,
        )
        return reward_manager
