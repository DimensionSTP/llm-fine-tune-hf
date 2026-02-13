from typing import Any, Optional, Union
import os

from omegaconf import DictConfig

import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from transformers import PreTrainedTokenizer, ProcessorMixin, PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download

from tqdm import tqdm

from .collate_fns import collate_fn_vlm


def build_test_dataloader(
    test_dataset: Dataset,
    config: DictConfig,
    num_workers: int,
    sampler: Optional[Sampler],
) -> DataLoader:
    return DataLoader(
        dataset=test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=collate_fn_vlm if config.modality != "text" else None,
    )


def generate_test_results(
    test_loader: DataLoader,
    model: PreTrainedModel,
    data_encoder: Union[PreTrainedTokenizer, ProcessorMixin],
    config: DictConfig,
    device: Union[int, torch.device],
    tqdm_desc: str,
    tqdm_disable: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(
            test_loader,
            desc=tqdm_desc,
            disable=tqdm_disable,
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                **config.generation_config,
            ).cpu()

            instructions = data_encoder.batch_decode(
                batch["input_ids"],
                skip_special_tokens=True,
            )

            generations = data_encoder.batch_decode(
                outputs[:, batch["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            labels = batch["labels"]

            for instruction, generation, label in zip(instructions, generations, labels):
                results.append(
                    {
                        "instruction": instruction,
                        "generation": generation,
                        "label": label,
                    }
                )

    return results


def save_test_results_json(
    results: list[dict[str, Any]],
    output_dir: str,
    output_name: str,
) -> pd.DataFrame:
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
    return df


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
        divisors = [d for d in range(1, test_gpu_count + 1) if test_gpu_count % d == 0]
        tp_size = min(divisors, key=lambda d: (abs(d - tp_size), -d))

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
    stop_token_ids: list[int],
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


def build_lora_request(
    config: DictConfig,
    lora_int_id: int,
) -> Optional[LoRARequest]:
    if not config.is_peft:
        return None

    return LoRARequest(
        lora_name=config.peft_test.adapter_name,
        lora_int_id=lora_int_id,
        lora_path=config.peft_test.adapter_path,
    )


def load_test_dataframe(
    config: DictConfig,
) -> pd.DataFrame:
    file_name = f"{config.dataset_name}.{config.dataset_format}"
    full_data_path = os.path.join(
        config.connected_dir,
        "data",
        config.test_data_dir,
        file_name,
    )

    if config.dataset_format == "parquet":
        df = pd.read_parquet(full_data_path)
    elif config.dataset_format in ["json", "jsonl"]:
        df = pd.read_json(
            full_data_path,
            lines=True if config.dataset_format == "jsonl" else False,
        )
    elif config.dataset_format in ["csv", "tsv"]:
        df = pd.read_csv(
            full_data_path,
            sep="\t" if config.dataset_format == "tsv" else None,
        )
    else:
        raise ValueError(f"Unsupported dataset format: {config.dataset_format}")

    df = df.fillna("_")
    return df
