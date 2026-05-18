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

import re
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)

from peft import PeftModel
from safetensors.torch import safe_open, save_file

import hydra
from omegaconf import DictConfig, OmegaConf


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


def config_as_bool(
    value: object,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return value
    if value is None:
        return False
    return str(value).lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def normalize_hf_max_shard_size(
    value: object,
) -> str:
    shard_size = str(value)
    replacements = {
        "GiB": "GB",
        "MiB": "MB",
        "KiB": "KB",
    }
    for source, target in replacements.items():
        if shard_size.endswith(source):
            return f"{shard_size[:-len(source)]}{target}"
    return shard_size


def parse_size_to_bytes(
    value: object,
) -> int:
    match = re.match(
        r"^([0-9]+(?:\.[0-9]+)?)([kmgt]?i?b?)?$",
        str(value).strip(),
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Invalid size: {value}")

    number = float(
        match.group(1),
    )
    suffix = (match.group(2) or "b").lower()
    units = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
        "k": 1024,
        "m": 1024**2,
        "g": 1024**3,
        "t": 1024**4,
    }
    if suffix not in units:
        raise ValueError(f"Invalid size suffix: {suffix}")
    return int(
        number * units[suffix],
    )


def load_safetensors_index(
    checkpoint_dir: Path,
) -> dict:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")
    with index_path.open() as f:
        return json.load(f)


def copy_non_weight_files(
    input_dir: Path,
    output_dir: Path,
) -> None:
    for source in input_dir.iterdir():
        if source.name == "model.safetensors.index.json":
            continue
        if source.suffix == ".safetensors":
            continue

        target = output_dir / source.name
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                dirs_exist_ok=True,
            )
        else:
            shutil.copy2(
                source,
                target,
            )


def _match_qwen_moe_expert_key(
    key: str,
) -> re.Match[str] | None:
    return re.match(
        r"^(?P<prefix>.+\.mlp\.experts)\."
        r"(?P<expert_idx>\d+)\."
        r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$",
        key,
    )


def group_qwen_moe_expert_keys(
    weight_map: dict[str, str],
) -> dict[str, dict[int, dict[str, str]]]:
    grouped_keys: dict[str, dict[int, dict[str, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for key in weight_map:
        match = _match_qwen_moe_expert_key(
            key,
        )
        if match is None:
            continue

        grouped_keys[match.group("prefix")][int(match.group("expert_idx"))][
            match.group("projection")
        ] = key

    return grouped_keys


def _load_safetensor_tensor(
    source_dir: Path,
    weight_map: dict[str, str],
    name: str,
) -> torch.Tensor:
    shard_name = weight_map[name]
    with safe_open(
        source_dir / shard_name,
        framework="pt",
        device="cpu",
    ) as shard:
        return shard.get_tensor(name)


class _SafetensorsShardWriter:
    def __init__(
        self,
        temp_dir: Path,
        max_shard_size_bytes: int,
    ) -> None:
        self.temp_dir = temp_dir
        self.max_shard_size_bytes = max_shard_size_bytes
        self.weight_map: dict[str, str] = {}
        self.temp_shards: list[Path] = []
        self.current_tensors: dict[str, torch.Tensor] = {}
        self.current_names: list[str] = []
        self.current_bytes = 0
        self.total_size = 0

    def add_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
    ) -> None:
        tensor_bytes = tensor.numel() * tensor.element_size()
        if (
            self.current_tensors
            and self.current_bytes + tensor_bytes > self.max_shard_size_bytes
        ):
            self.flush()

        self.current_tensors[name] = tensor
        self.current_names.append(name)
        self.current_bytes += tensor_bytes
        self.total_size += tensor_bytes

        if tensor_bytes >= self.max_shard_size_bytes:
            self.flush()

    def flush(
        self,
    ) -> None:
        if not self.current_tensors:
            return

        temp_name = f"model-{len(self.temp_shards) + 1:05d}.safetensors"
        temp_path = self.temp_dir / temp_name
        save_file(
            self.current_tensors,
            temp_path,
            metadata={
                "format": "pt",
            },
        )
        for name in self.current_names:
            self.weight_map[name] = temp_name

        print(
            f"[pack-qwen-moe] wrote {temp_name}: "
            f"{self.current_bytes / (1024 ** 3):.2f} GiB, "
            f"{len(self.current_names)} tensors",
            flush=True,
        )
        self.temp_shards.append(temp_path)
        self.current_tensors = {}
        self.current_names = []
        self.current_bytes = 0

    def finalize_shards(
        self,
    ) -> dict[str, str]:
        self.flush()

        shard_count = len(self.temp_shards)
        final_name_map: dict[str, str] = {}
        for idx, temp_path in enumerate(
            self.temp_shards,
            start=1,
        ):
            final_name = f"model-{idx:05d}-of-{shard_count:05d}.safetensors"
            final_path = self.temp_dir / final_name
            temp_path.rename(final_path)
            final_name_map[temp_path.name] = final_name

        return {
            tensor_name: final_name_map[temp_name]
            for tensor_name, temp_name in self.weight_map.items()
        }


def _validate_qwen_moe_expert_group(
    prefix: str,
    expert_keys: dict[int, dict[str, str]],
) -> list[int]:
    expert_indices = sorted(expert_keys)
    if expert_indices != list(
        range(len(expert_indices)),
    ):
        raise ValueError(
            f"Non-contiguous Qwen MoE expert indices for {prefix}: "
            f"{expert_indices[:3]} ... {expert_indices[-3:]}"
        )

    for expert_idx in expert_indices:
        missing = {
            "gate_proj",
            "up_proj",
            "down_proj",
        } - set(expert_keys[expert_idx])
        if missing:
            raise ValueError(
                f"Missing Qwen MoE expert weights for {prefix}.{expert_idx}: "
                f"{sorted(missing)}"
            )

    return expert_indices


def _pack_qwen_moe_expert_group(
    prefix: str,
    expert_keys: dict[int, dict[str, str]],
    source_dir: Path,
    weight_map: dict[str, str],
    shard_writer: _SafetensorsShardWriter,
) -> None:
    expert_indices = _validate_qwen_moe_expert_group(
        prefix=prefix,
        expert_keys=expert_keys,
    )

    first_keys = expert_keys[expert_indices[0]]
    first_gate = _load_safetensor_tensor(
        source_dir=source_dir,
        weight_map=weight_map,
        name=first_keys["gate_proj"],
    )
    first_up = _load_safetensor_tensor(
        source_dir=source_dir,
        weight_map=weight_map,
        name=first_keys["up_proj"],
    )
    first_down = _load_safetensor_tensor(
        source_dir=source_dir,
        weight_map=weight_map,
        name=first_keys["down_proj"],
    )
    if first_gate.shape != first_up.shape:
        raise ValueError(
            f"gate_proj and up_proj shapes differ for {prefix}.0: "
            f"{tuple(first_gate.shape)} vs {tuple(first_up.shape)}"
        )

    gate_rows, hidden_size = first_gate.shape
    down_rows, down_cols = first_down.shape
    gate_up_proj = torch.empty(
        (
            len(expert_indices),
            gate_rows * 2,
            hidden_size,
        ),
        dtype=first_gate.dtype,
        device=first_gate.device,
    )
    down_proj = torch.empty(
        (
            len(expert_indices),
            down_rows,
            down_cols,
        ),
        dtype=first_down.dtype,
        device=first_down.device,
    )

    for output_idx, expert_idx in enumerate(expert_indices):
        keys = expert_keys[expert_idx]
        gate = _load_safetensor_tensor(
            source_dir=source_dir,
            weight_map=weight_map,
            name=keys["gate_proj"],
        )
        up = _load_safetensor_tensor(
            source_dir=source_dir,
            weight_map=weight_map,
            name=keys["up_proj"],
        )
        down = _load_safetensor_tensor(
            source_dir=source_dir,
            weight_map=weight_map,
            name=keys["down_proj"],
        )

        if gate.shape != first_gate.shape or up.shape != first_up.shape:
            raise ValueError(
                f"Inconsistent gate/up shape for {prefix}.{expert_idx}: "
                f"gate={tuple(gate.shape)}, up={tuple(up.shape)}"
            )
        if down.shape != first_down.shape:
            raise ValueError(
                f"Inconsistent down shape for {prefix}.{expert_idx}: "
                f"{tuple(down.shape)}"
            )

        gate_up_proj[output_idx, :gate_rows, :].copy_(gate)
        gate_up_proj[output_idx, gate_rows:, :].copy_(up)
        down_proj[output_idx].copy_(down)

    shard_writer.add_tensor(
        f"{prefix}.gate_up_proj",
        gate_up_proj,
    )
    shard_writer.add_tensor(
        f"{prefix}.down_proj",
        down_proj,
    )
    shard_writer.flush()


def pack_qwen_moe_experts_checkpoint(
    checkpoint_dir: str,
    max_shard_size: object,
) -> int:
    """Rewrite a saved Qwen MoE checkpoint with fused expert tensors."""
    source_dir = Path(checkpoint_dir)
    index = load_safetensors_index(
        checkpoint_dir=source_dir,
    )
    old_weight_map: dict[str, str] = index["weight_map"]
    expert_groups = group_qwen_moe_expert_keys(
        weight_map=old_weight_map,
    )
    if not expert_groups:
        return 0

    max_shard_size_bytes = parse_size_to_bytes(
        max_shard_size,
    )
    temp_dir = source_dir.with_name(f".{source_dir.name}.packed-{os.getpid()}")
    backup_dir = source_dir.with_name(f".{source_dir.name}.unpacked-{os.getpid()}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    temp_dir.mkdir(
        parents=True,
        exist_ok=False,
    )
    copy_non_weight_files(
        input_dir=source_dir,
        output_dir=temp_dir,
    )

    shard_to_names: dict[str, list[str]] = defaultdict(list)
    for tensor_name, shard_name in old_weight_map.items():
        shard_to_names[shard_name].append(tensor_name)

    old_expert_keys = {
        key
        for expert_keys in expert_groups.values()
        for projections in expert_keys.values()
        for key in projections.values()
    }
    shard_writer = _SafetensorsShardWriter(
        temp_dir=temp_dir,
        max_shard_size_bytes=max_shard_size_bytes,
    )

    for shard_name in sorted(shard_to_names):
        with safe_open(
            source_dir / shard_name,
            framework="pt",
            device="cpu",
        ) as shard:
            for tensor_name in shard_to_names[shard_name]:
                if tensor_name in old_expert_keys:
                    continue
                shard_writer.add_tensor(
                    tensor_name,
                    shard.get_tensor(tensor_name),
                )

    shard_writer.flush()

    for prefix in sorted(expert_groups):
        _pack_qwen_moe_expert_group(
            prefix=prefix,
            expert_keys=expert_groups[prefix],
            source_dir=source_dir,
            weight_map=old_weight_map,
            shard_writer=shard_writer,
        )

    final_weight_map = shard_writer.finalize_shards()
    if any(
        _match_qwen_moe_expert_key(
            key,
        )
        is not None
        for key in final_weight_map
    ):
        raise RuntimeError(
            "Qwen MoE checkpoint still contains unpacked per-expert keys"
        )

    metadata = dict(index.get("metadata") or {})
    metadata["total_size"] = shard_writer.total_size
    with (temp_dir / "model.safetensors.index.json").open("w") as f:
        json.dump(
            {
                "metadata": metadata,
                "weight_map": final_weight_map,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    source_dir.rename(backup_dir)
    try:
        temp_dir.rename(source_dir)
    except Exception:
        backup_dir.rename(source_dir)
        raise
    shutil.rmtree(backup_dir)

    return len(expert_groups)


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
    save_pretrained_kwargs = {
        "safe_serialization": True,
    }
    merge_max_shard_size = OmegaConf.select(
        config,
        "merge_max_shard_size",
        default=None,
    )
    if merge_max_shard_size is not None:
        save_pretrained_kwargs["max_shard_size"] = normalize_hf_max_shard_size(
            merge_max_shard_size
        )

    pack_qwen_moe_experts = config_as_bool(
        OmegaConf.select(
            config,
            "merge_pack_qwen_moe_experts",
            default=False,
        )
    )
    packed_expert_groups = 0

    merged_model.save_pretrained(
        merged_output_path,
        **save_pretrained_kwargs,
    )
    if pack_qwen_moe_experts:
        packed_expert_groups = pack_qwen_moe_experts_checkpoint(
            checkpoint_dir=merged_output_path,
            max_shard_size=save_pretrained_kwargs.get(
                "max_shard_size",
                "5GB",
            ),
        )

    print(f"[OK] Merged LoRA (and modules_to_save if present) into base model.")
    print(f"[OK] Base model: {base_model_name_or_path}")
    print(f"[OK] Adapter checkpoint: {adapter_path}")
    print(f"[OK] Saved merged model to: {merged_output_path}")
    if merge_max_shard_size is not None:
        print(f"[OK] max_shard_size: {merge_max_shard_size}")
    if pack_qwen_moe_experts:
        print(f"[OK] packed Qwen MoE expert groups: {packed_expert_groups}")


if __name__ == "__main__":
    merge_lora()
