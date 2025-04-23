from typing import Tuple, Dict, List, Any, Optional, Union

import os
import json

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import ModuleList

from transformers import AutoModel, AutoTokenizer

from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from transformers.generation import GenerationMixin

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaModel,
    LlamaForCausalLM,
    KwargsForCausalLM,
)

from safetensors.torch import save_file, load_file

from huggingface_hub import hf_hub_download

from .moff import MixtureOfFeedForward


class MixLlamaForCausalLM(LlamaForCausalLM, GenerationMixin):
    def __init__(
        self,
        configs: Tuple[LlamaConfig, ...],
        pretrained_models: Optional[List[LlamaForCausalLM]] = None,
    ) -> None:
        super().__init__(configs[0])
        self.post_init()

        self.configs = configs
        self.models = ModuleList([LlamaModel(config) for config in configs])
        self.total_hidden_size = sum(config.hidden_size for config in configs)
        self.lm_head = nn.Linear(
            self.total_hidden_size,
            configs[0].vocab_size,
            bias=False,
            dtype=configs[0].torch_dtype,
        )

        self.moff = MixtureOfFeedForward(configs)

        if pretrained_models is not None:
            for i, pretrained in enumerate(pretrained_models):
                self.models[i].embed_tokens.weight.data.copy_(
                    pretrained.embed_tokens.weight.data
                )
                self.models[i].norm.load_state_dict(pretrained.norm.state_dict())
                self.models[i].rotary_emb.load_state_dict(
                    pretrained.rotary_emb.state_dict()
                )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.configs[0].output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.configs[0].output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.configs[0].use_return_dict
        )

        branch_hidden_states = []
        branch_past = []
        branch_attentions = []

        for model in self.models:
            outputs: BaseModelOutputWithPast = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = outputs.last_hidden_state
            if hidden_states.dim() < 3:
                hidden_states = hidden_states.unsqueeze(0)
            branch_hidden_states.append(hidden_states)

            if use_cache:
                past = outputs.past_key_values if return_dict else outputs[1]
                if past is not None:
                    branch_past.append(past)
            if output_attentions:
                attn = outputs.attentions if return_dict else None
                if attn is not None:
                    branch_attentions.append(attn)

        combined_hidden = self.moff(tuple(branch_hidden_states))

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(combined_hidden[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def state_dict(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Override state_dict to include MOFF weights."""
        state_dict = super().state_dict(*args, **kwargs)
        moff_state_dict = self.moff.state_dict()
        for key, value in moff_state_dict.items():
            state_dict[f"moff.{key}"] = value
        return state_dict

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        """Override load_state_dict to handle MOFF weights."""
        moff_state_dict = {}
        for key in list(state_dict.keys()):
            if key.startswith("moff."):
                moff_state_dict[key[5:]] = state_dict.pop(key)

        if moff_state_dict:
            self.moff.load_state_dict(moff_state_dict)

        return super().load_state_dict(
            state_dict,
            strict,
        )

    def save_pretrained(
        self,
        save_directory: str,
        **kwargs: Any,
    ) -> None:
        """Save the model weights and configuration to a directory.

        This will save all weights in a single file (model.safetensors),
        and save the configuration in config.json.
        """
        os.makedirs(
            save_directory,
            exist_ok=True,
        )

        state_dict = {}

        for i, model in enumerate(self.models):
            model_state_dict = model.state_dict()
            for key, value in model_state_dict.items():
                state_dict[f"model{i}.{key}"] = value

        moff_state_dict = self.moff.state_dict()
        for key, value in moff_state_dict.items():
            state_dict[f"moff.{key}"] = value

        lm_head_state_dict = self.lm_head.state_dict()
        for key, value in lm_head_state_dict.items():
            state_dict[f"lm_head.{key}"] = value

        model_path = os.path.join(
            save_directory,
            "model.safetensors",
        )
        save_file(
            state_dict,
            model_path,
        )

        config = {
            "model_configs": [config.to_dict() for config in self.configs],
            "num_models": len(self.models),
        }
        config_path = os.path.join(
            save_directory,
            "config.json",
        )
        with open(config_path, "w") as f:
            json.dump(
                config,
                f,
                indent=2,
            )

        tokenizer = AutoTokenizer.from_pretrained(self.configs[0]._name_or_path)
        tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        model_names: Optional[List[str]] = None,
        initialize: bool = False,
        *args: Any,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs: Any,
    ) -> "MixLlamaForCausalLM":
        """Load a model from a directory or initialize with external models.

        Args:
            pretrained_model_name_or_path (str, optional): Path to the saved model directory, or model identifier from huggingface.co/models
            model_names (List[str], optional): Names of external models to load when initialize=True
            initialize (bool): If True, initialize with external models.
                             If False, load from saved files in pretrained_model_name_or_path.
            cache_dir (str, optional): Path to a directory in which a downloaded pretrained model should be cached.
            force_download (bool, optional): Whether to force the (re-)download of the model weights and configuration files.
            local_files_only (bool, optional): Whether to only look at local files.
            token (str or bool, optional): The token to use as HTTP bearer authorization for remote files.
            revision (str, optional): The specific model version to use. It can be a branch name, a tag name, or a commit id.
            *args, **kwargs: Additional arguments passed to AutoModel.from_pretrained
        """
        if initialize:
            if model_names is None:
                raise ValueError("model_names must be provided when initialize=True")

            pretrained_models = [
                AutoModel.from_pretrained(
                    name,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    *args,
                    **kwargs,
                )
                for name in model_names
            ]
            configs = [model.config for model in pretrained_models]

            model = cls(
                configs=configs,
                pretrained_models=pretrained_models,
            )

            for i, pretrained in enumerate(pretrained_models):
                model.models[i].embed_tokens.weight.data.copy_(
                    pretrained.embed_tokens.weight.data
                )
                model.models[i].norm.load_state_dict(pretrained.norm.state_dict())
                model.models[i].rotary_emb.load_state_dict(
                    pretrained.rotary_emb.state_dict()
                )
        else:
            if pretrained_model_name_or_path is None:
                raise ValueError(
                    "pretrained_model_name_or_path must be provided when initialize=False"
                )

            if os.path.isdir(pretrained_model_name_or_path):
                config_path = os.path.join(
                    pretrained_model_name_or_path,
                    "config.json",
                )
                weights_path = os.path.join(
                    pretrained_model_name_or_path,
                    "model.safetensors",
                )
            else:
                config_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="config.json",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                )
                weights_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="model.safetensors",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                )

            with open(config_path, "r") as f:
                config = json.load(f)

            configs = [LlamaConfig.from_dict(cfg) for cfg in config["model_configs"]]
            model = cls(configs=configs)

            state_dict = load_file(weights_path)

            for i in range(config["num_models"]):
                model_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith(f"model{i}."):
                        model_state_dict[key[len(f"model{i}.") :]] = value
                model.models[i].load_state_dict(model_state_dict)

            moff_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("moff."):
                    moff_state_dict[key[5:]] = value
            model.moff.load_state_dict(moff_state_dict)

            lm_head_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("lm_head."):
                    lm_head_state_dict[key[8:]] = value
            model.lm_head.load_state_dict(lm_head_state_dict)

        return model

    @classmethod
    def _load_state_dict(
        cls,
        model_name: str,
        *args,
        **kwargs,
    ):
        model = AutoModel.from_pretrained(
            model_name,
            *args,
            **kwargs,
        )
        return model.state_dict()
