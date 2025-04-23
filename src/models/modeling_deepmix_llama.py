from typing import Tuple, List, Any, Optional, Union

import math
import os
import json

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import ModuleList

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from transformers.generation import GenerationMixin

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
    LlamaRMSNorm,
    LlamaForCausalLM,
    KwargsForCausalLM,
)

from safetensors.torch import save_file, load_file

from huggingface_hub import hf_hub_download

from .moff import MixtureOfInterFeedForward


class DeepMixLlamaModel(nn.Module):
    def __init__(
        self,
        configs: Tuple[LlamaConfig, ...],
        mix_interval: int,
    ):
        super().__init__()
        self.configs = configs
        self.mix_interval = mix_interval
        self.num_models = len(configs)
        self.num_layers = configs[0].num_hidden_layers

        self.embeddings = ModuleList(
            [
                nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
                for config in configs
            ]
        )

        self.embed_projs = ModuleList([nn.Identity() for _ in configs])

        self.decoders = ModuleList(
            [
                ModuleList(
                    [
                        LlamaDecoderLayer(
                            config=config,
                            layer_idx=layer_idx,
                        )
                        for layer_idx in range(config.num_hidden_layers)
                    ]
                )
                for config in configs
            ]
        )

        self.rotary_embs = ModuleList(
            [LlamaRotaryEmbedding(config=config) for config in configs]
        )

        self.final_norms = ModuleList(
            [
                LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                for config in configs
            ]
        )

        num_blocks = math.ceil(self.num_layers / self.mix_interval)
        num_moffs = max(0, num_blocks - 1)
        self.moffs = ModuleList(
            [MixtureOfInterFeedForward(configs) for _ in range(num_moffs)]
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Union[Cache, List[torch.FloatTensor]]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[BaseModelOutputWithPast, Tuple]:
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
        use_cache = use_cache if use_cache is not None else self.configs[0].use_cache
        return_dict = (
            return_dict if return_dict is not None else self.configs[0].use_return_dict
        )

        if inputs_embeds is None:
            hs = [
                self.embed_projs[i](self.embeddings[i](input_ids))
                for i in range(self.num_models)
            ]
        else:
            hs = [inputs_embeds for _ in range(self.num_models)]

        batch_size, seq_len, _ = hs[0].size()
        device = hs[0].device
        dtype = hs[0].dtype

        if position_ids is None:
            if past_key_values is not None and past_key_values[0] is not None:
                past_length = past_key_values[0].get_seq_length()
                position_ids = (
                    torch.arange(
                        past_length,
                        past_length + seq_len,
                        device=device,
                    )
                    .unsqueeze(0)
                    .expand(
                        batch_size,
                        seq_len,
                    )
                )
            else:
                position_ids = (
                    torch.arange(
                        seq_len,
                        device=device,
                    )
                    .unsqueeze(0)
                    .expand(
                        batch_size,
                        seq_len,
                    )
                )

        causal_mask = (
            torch.triu(
                torch.full(
                    (
                        seq_len,
                        seq_len,
                    ),
                    -1e9,
                    device=device,
                    dtype=dtype,
                ),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(
                batch_size,
                1,
                seq_len,
                seq_len,
            )
        )

        if attention_mask is not None:
            attn_mask = (1.0 - attention_mask.to(dtype=hs[0].dtype)) * -1e9
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask + attn_mask

        all_hidden_states = [] if output_hidden_states else None
        moff_idx = 0

        if use_cache:
            if past_key_values is None:
                past_key_values = [DynamicCache() for _ in range(self.num_models)]

        if (
            cache_position is None
            and past_key_values is not None
            and past_key_values[0] is not None
        ):
            past_length = past_key_values[0].get_seq_length()
            cache_position = torch.arange(
                past_length,
                past_length + seq_len,
                device=device,
            )

        for layer_idx in range(self.num_layers):
            new_hs = []
            for branch_idx in range(self.num_models):
                position_embeddings = self.rotary_embs[branch_idx](
                    hs[branch_idx], position_ids
                )

                layer_output = self.decoders[branch_idx][layer_idx](
                    hs[branch_idx],
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values[branch_idx] if use_cache else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                new_hs.append(layer_output[0])
            hs = new_hs

            if (layer_idx + 1) % self.mix_interval == 0 and (
                layer_idx + 1
            ) < self.num_layers:
                hs = list(self.moffs[moff_idx](tuple(hs)))
                moff_idx += 1

            if output_hidden_states:
                all_hidden_states.append(hs)

        final_hs = [norm(h) for norm, h in zip(self.final_norms, hs)]
        concat_h = torch.cat(
            final_hs,
            dim=-1,
        )

        return BaseModelOutputWithPast(
            last_hidden_state=concat_h,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class DeepMixLlamaForCausalLM(LlamaForCausalLM, GenerationMixin):
    def __init__(
        self,
        configs: Tuple[LlamaConfig, ...],
        mix_interval: int,
    ):
        super().__init__(configs[0])
        self.post_init()

        self.configs = configs
        self.model = DeepMixLlamaModel(
            configs=configs,
            mix_interval=mix_interval,
        )
        total_hidden_size = sum(config.hidden_size for config in configs)
        self.lm_head = nn.Linear(
            total_hidden_size,
            configs[0].vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
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

        outputs = self.model(
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
        hidden_states = hidden_states.to(dtype=self.configs[0].torch_dtype)

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

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

        for i, model in enumerate(self.model.embeddings):
            model_state_dict = model.state_dict()
            for key, value in model_state_dict.items():
                state_dict[f"model{i}.{key}"] = value

        for i, moff in enumerate(self.model.moffs):
            moff_state_dict = moff.state_dict()
            for key, value in moff_state_dict.items():
                state_dict[f"moff{i}.{key}"] = value

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
            "num_models": len(self.model.embeddings),
            "mix_interval": self.model.mix_interval,
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
        mix_interval: Optional[int] = None,
        *args: Any,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs: Any,
    ) -> "DeepMixLlamaForCausalLM":
        """Load a model from a directory or initialize with external models.

        Args:
            pretrained_model_name_or_path (str, optional): Path to the saved model directory, or model identifier from huggingface.co/models
            model_names (List[str], optional): Names of external models to load when initialize=True
            initialize (bool): If True, initialize with external models.
                             If False, load from saved files in pretrained_model_name_or_path.
            mix_interval (int, optional): Mixing interval between MOFF layers. Required when initialize=True.
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
            if mix_interval is None:
                raise ValueError("mix_interval must be provided when initialize=True")

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
                mix_interval=mix_interval,
            )

            for i, pretrained in enumerate(pretrained_models):
                model.model.embeddings[i].weight.data.copy_(
                    pretrained.embed_tokens.weight.data
                )
                model.model.final_norms[i].load_state_dict(pretrained.norm.state_dict())
                model.model.rotary_embs[i].load_state_dict(
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
            model = cls(
                configs=configs,
                mix_interval=config["mix_interval"],
            )

            state_dict = load_file(weights_path)

            for i in range(config["num_models"]):
                model_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith(f"model{i}."):
                        model_state_dict[key[len(f"model{i}.") :]] = value
                model.model.embeddings[i].load_state_dict(model_state_dict)

            for i in range(len(model.model.moffs)):
                moff_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith(f"moff{i}."):
                        moff_state_dict[key[len(f"moff{i}.") :]] = value
                if moff_state_dict:
                    model.model.moffs[i].load_state_dict(moff_state_dict)

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
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            *args,
            **kwargs,
        )
        return model.state_dict()
