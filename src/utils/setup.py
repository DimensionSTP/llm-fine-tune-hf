from typing import Dict, Union, Optional, Any
import os

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import importlib

datasets = importlib.import_module("datasets")
HFDataset = datasets.Dataset

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from ..datasets import *


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
        dataset: Union[DPOStructuralDataset, DPOConversationalDataset] = instantiate(
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

        model = AutoModelForCausalLM.from_pretrained(
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

            return model

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=self.config.gradient_checkpointing_kwargs,
            )

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
            peft_config = LoraConfig(**peft_config_dict)
            model = get_peft_model(
                model=model,
                peft_config=peft_config,
            )

        return model

    def get_data_encoder(self) -> PreTrainedTokenizer:
        if self.config.is_preprocessed:
            data_encoder_path = self.config.custom_data_encoder_path
        else:
            data_encoder_path = self.config.pretrained_model_name

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
