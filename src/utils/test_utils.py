from typing import Dict, List, Union, Optional, Any
import os
import json

from omegaconf import DictConfig, OmegaConf

import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from transformers import PreTrainedTokenizerBase, ProcessorMixin, PreTrainedModel

from vllm import LLM, SamplingParams

from huggingface_hub import snapshot_download

from tqdm import tqdm

from .collate_fns import collate_fn_vlm
from ..helpers.dataset_paths import resolve_dataset_file_specs
from .metadata_security import redact_metadata_payload


def build_test_dataloader(
    test_dataset: Dataset,
    config: DictConfig,
    dataloader_kwargs: Dict[str, Any],
    sampler: Optional[Sampler],
) -> DataLoader:
    return DataLoader(
        dataset=test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_fn_vlm if config.modality != "text" else None,
        **dataloader_kwargs,
    )


def generate_test_results(
    test_loader: DataLoader,
    model: PreTrainedModel,
    data_encoder: Union[PreTrainedTokenizerBase, ProcessorMixin],
    config: DictConfig,
    device: Union[int, torch.device],
    tqdm_desc: str,
    tqdm_disable: bool,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(
            test_loader,
            desc=tqdm_desc,
            disable=tqdm_disable,
        ):
            generation_inputs = build_generation_inputs(
                batch=batch,
                device=device,
            )

            outputs = model.generate(
                **generation_inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                **config.generation_config,
            ).cpu()

            text_encoder = resolve_text_encoder(data_encoder=data_encoder)
            instructions = text_encoder.batch_decode(
                batch["input_ids"],
                skip_special_tokens=True,
            )

            generations = text_encoder.batch_decode(
                outputs[:, batch["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            labels = batch["labels"]

            for instruction, generation, label in zip(
                instructions, generations, labels
            ):
                results.append(
                    {
                        "instruction": instruction,
                        "generation": generation,
                        "label": label,
                    }
                )

    return results


def build_generation_inputs(
    batch: Dict[str, Any],
    device: Union[int, torch.device],
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key != "labels"
        and isinstance(
            value,
            torch.Tensor,
        )
    }


def resolve_text_encoder(
    data_encoder: Union[PreTrainedTokenizerBase, ProcessorMixin],
) -> PreTrainedTokenizerBase:
    if isinstance(data_encoder, PreTrainedTokenizerBase):
        return data_encoder
    return data_encoder.tokenizer


def save_test_results_json(
    results: List[Dict[str, Any]],
    output_dir: str,
    output_name: str,
) -> Dict[str, Any]:
    os.makedirs(
        output_dir,
        exist_ok=True,
    )
    test_output_path = os.path.join(
        output_dir,
        f"{output_name}.json",
    )

    df = pd.DataFrame(results)
    df.to_json(
        test_output_path,
        orient="records",
        indent=2,
        force_ascii=False,
    )
    return {
        "dataframe": df,
        "result_path": test_output_path,
    }


def write_inference_manifest(
    config: DictConfig,
    result_path: str,
    sampling_params: Optional[SamplingParams],
    tp_size: Optional[int],
    vision_patch_embedding_result: Optional[Dict[str, Any]],
    vllm_lora_runtime: Optional[Dict[str, Any]],
) -> str:
    if not os.path.isfile(result_path):
        raise FileNotFoundError(f"Inference result not found: {result_path}")

    test_dataset_files = resolve_dataset_file_specs(
        dataset_name=config.dataset_name,
        dataset_format=config.dataset_format,
        data_path=config.data_path,
        dataset_subdir=config.test_dataset_subdir,
        dataset_file_path=config.test_dataset_file_path,
        dataset_file_paths=config.test_dataset_file_paths,
        dataset_files=config.test_dataset_files,
        allow_dataset_file_name_mismatch=config.allow_test_dataset_file_name_mismatch,
        path_label="test_dataset",
        allow_weight=False,
    )
    active_data_encoder_path = (
        config.custom_data_encoder_path
        if config.is_preprocessed
        else config.pretrained_model_name
    )
    result_stem = os.path.splitext(result_path)[0]
    manifest_path = f"{result_stem}_manifest.json"
    runtime = _build_inference_runtime_section(
        config=config,
        sampling_params=sampling_params,
        tp_size=tp_size,
    )
    if vision_patch_embedding_result is not None:
        runtime["vision_patch_embedding"] = vision_patch_embedding_result
    resolved_input = {
        "active_data_encoder_path": active_data_encoder_path,
        "dataset_files": test_dataset_files,
    }
    if vllm_lora_runtime is not None:
        resolved_input["peft_adapter"] = {
            "weights_sha256": vllm_lora_runtime["source_weights_sha256"],
            "config_sha256": vllm_lora_runtime["source_config_sha256"],
        }
        runtime["vllm_lora_adapter"] = {
            "effective_adapter_path": vllm_lora_runtime["effective_adapter_path"],
            "effective_weights_sha256": vllm_lora_runtime["effective_weights_sha256"],
            "total_tensor_count": vllm_lora_runtime["total_tensor_count"],
            "remapped_tensor_count": vllm_lora_runtime["remapped_tensor_count"],
            "passthrough_tensor_count": vllm_lora_runtime["passthrough_tensor_count"],
            "action": vllm_lora_runtime["action"],
        }
    manifest = redact_metadata_payload(
        config=config,
        payload={
            "resolved_config": OmegaConf.to_container(
                config,
                resolve=True,
            ),
            "resolved_input": resolved_input,
            "runtime": runtime,
        },
    )
    temp_path = f"{manifest_path}.tmp.{os.getpid()}"
    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
    os.replace(
        temp_path,
        manifest_path,
    )
    return manifest_path


def resolve_vllm_tp_size(
    config: DictConfig,
    num_gpus: int,
) -> int:
    devices_limit = num_gpus
    if config.devices is not None:
        if isinstance(config.devices, int):
            devices_limit = config.devices
        elif isinstance(config.devices, str):
            devices_limit = len(
                [d for d in config.devices.split(",") if d.strip() != ""]
            )
        elif isinstance(config.devices, list):
            devices_limit = len(config.devices)

    test_gpu_count = devices_limit
    if config.test_vllm.gpu_count is not None:
        test_gpu_count = int(config.test_vllm.gpu_count)

    if test_gpu_count > devices_limit:
        test_gpu_count = devices_limit
    if test_gpu_count < 1:
        test_gpu_count = 1

    tp_size = int(config.test_vllm.tp_size)
    if tp_size > test_gpu_count:
        tp_size = test_gpu_count
    if test_gpu_count % tp_size != 0:
        divisors = [
            d
            for d in range(
                1,
                test_gpu_count + 1,
            )
            if test_gpu_count % d == 0
        ]
        tp_size = min(
            divisors,
            key=lambda d: (abs(d - tp_size), -d),
        )

    return tp_size


def build_vllm(
    config: DictConfig,
    tp_size: int,
) -> LLM:
    try:
        llm = LLM(
            model=config.pretrained_model_name,
            tokenizer=config.pretrained_model_name,
            revision=config.revision,
            tensor_parallel_size=tp_size,
            seed=config.seed,
            trust_remote_code=True,
            max_model_len=config.max_length,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_lora=config.is_peft,
            max_lora_rank=config.peft_config.r,
        )
    except Exception:
        model_path = snapshot_download(
            repo_id=config.pretrained_model_name,
            revision=config.revision,
        )
        llm = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=tp_size,
            seed=config.seed,
            trust_remote_code=True,
            max_model_len=config.max_length,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_lora=config.is_peft,
            max_lora_rank=config.peft_config.r,
        )
    return llm


def build_sampling_params(
    config: DictConfig,
    stop_token_ids: List[int],
) -> SamplingParams:
    if config.do_sample:
        generation_config = config.generation_config
    else:
        generation_config = {
            "temperature": 0,
            "top_p": 1,
        }

    return SamplingParams(
        max_tokens=config.max_new_tokens,
        skip_special_tokens=True,
        stop_token_ids=stop_token_ids,
        stop=[
            "### End",
            "\n### End",
        ],
        **generation_config,
    )


def load_test_dataframe(
    config: DictConfig,
) -> pd.DataFrame:
    data_specs = resolve_dataset_file_specs(
        dataset_name=config.dataset_name,
        dataset_format=config.dataset_format,
        data_path=config.data_path,
        dataset_subdir=config.test_dataset_subdir,
        dataset_file_path=config.test_dataset_file_path,
        dataset_file_paths=config.test_dataset_file_paths,
        dataset_files=config.test_dataset_files,
        allow_dataset_file_name_mismatch=config.allow_test_dataset_file_name_mismatch,
        path_label="test_dataset",
        allow_weight=False,
    )

    frames = []
    for data_spec in data_specs:
        if data_spec["format"] == "parquet":
            frame = pd.read_parquet(data_spec["path"])
        elif data_spec["format"] in ["json", "jsonl"]:
            frame = pd.read_json(
                data_spec["path"],
                lines=True if data_spec["format"] == "jsonl" else False,
            )
        elif data_spec["format"] in ["csv", "tsv"]:
            frame = pd.read_csv(
                data_spec["path"],
                sep="\t" if data_spec["format"] == "tsv" else ",",
            )
        else:
            raise ValueError(f"Unsupported dataset format: {data_spec['format']}")
        frames.append(frame)

    if len(frames) == 1:
        df = frames[0]
    else:
        df = pd.concat(
            frames,
            ignore_index=True,
        )
    df = df.fillna("_")
    return df


def _build_inference_runtime_section(
    config: DictConfig,
    sampling_params: Optional[SamplingParams],
    tp_size: Optional[int],
) -> Dict[str, Any]:
    if sampling_params is not None:
        return {
            "backend": "vllm",
            "device_map": None,
            "tensor_parallel_size": tp_size,
            "generation": _build_inference_generation_section(
                config=config,
                sampling_params=sampling_params,
            ),
        }

    device_map = (
        config.model_loading.inference.test_large_device_map
        if config.mode == "test_large"
        else config.model_loading.inference.device_map
    )
    if OmegaConf.is_config(device_map):
        device_map = OmegaConf.to_container(
            device_map,
            resolve=True,
        )
    return {
        "backend": "transformers",
        "device_map": device_map,
        "tensor_parallel_size": None,
        "generation": _build_inference_generation_section(
            config=config,
            sampling_params=sampling_params,
        ),
    }


def _build_inference_generation_section(
    config: DictConfig,
    sampling_params: Optional[SamplingParams],
) -> Dict[str, Any]:
    if sampling_params is None:
        configured_generation = OmegaConf.to_container(
            config.generation_config,
            resolve=True,
        )
        return {
            "seed": config.seed,
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            **configured_generation,
        }

    return {
        "seed": sampling_params.seed,
        "max_tokens": sampling_params.max_tokens,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "stop": sampling_params.stop,
        "stop_token_ids": sampling_params.stop_token_ids,
        "skip_special_tokens": sampling_params.skip_special_tokens,
    }
