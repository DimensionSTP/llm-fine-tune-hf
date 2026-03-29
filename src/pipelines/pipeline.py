import os

from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

import json

import pandas as pd

import torch
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler

from transformers import set_seed

import wandb

from tqdm import tqdm

from ..utils import *


def train(
    config: DictConfig,
) -> None:
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        wandb.init(
            project=config.project_name,
            name=config.logging_name,
        )

    if "seed" in config:
        set_seed(config.seed)

    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if (not is_distributed) and (config.devices is not None):
        if isinstance(config.devices, int):
            num_gpus = min(config.devices, torch.cuda.device_count())
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))
        elif isinstance(config.devices, str):
            os.environ["CUDA_VISIBLE_DEVICES"] = config.devices
        elif isinstance(config.devices, list):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.devices))

    setup = SetUp(config)

    if config.fine_tune_method == "sft":
        train_dataset = setup.get_train_dataset()
        val_dataset = setup.get_val_dataset() if config.use_validation else None
    else:
        train_dataset = setup.get_dataset()["train"]
        val_dataset = setup.get_dataset()["val"]

    model = setup.get_model()
    data_encoder = setup.get_data_encoder()

    training_arguments = setup.get_training_arguments()

    ds_config = setup.get_ds_config()
    if ds_config:
        training_arguments.deepspeed = ds_config

    trainer_config = OmegaConf.to_container(
        config.trainer,
        resolve=True,
    )
    trainer_config.pop(
        "_target_",
        None,
    )

    TrainerClass = get_class(config.trainer._target_)

    if config.fine_tune_method == "grpo":
        reward_manager = setup.get_reward_manager()
        trainer_config["reward_funcs"] = reward_manager.get_reward_funcs()

    if config.fine_tune_method == "gkd":
        trainer_config["teacher_model"] = config.teacher.model

    trainer = TrainerClass(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=data_encoder,
        **trainer_config,
    )
    if patch_qwen_packed_moe_vllm_sync(
        trainer=trainer,
        config=config,
    ):
        print("[patch] Applied Qwen packed-MoE vLLM sync filter for router-with-lora GRPO.")
    elif patch_sparse_decoder_moe_vllm_sync(
        trainer=trainer,
        config=config,
    ):
        print(
            "[patch] Applied sparse-decoder MoE vLLM sync filter for router-with-lora GRPO."
        )

    try:
        trainer.train(
            resume_from_checkpoint=(
                config.resume_from_checkpoint if config.resume_training else None
            )
        )
        trainer.save_model()

        if rank == 0:
            wandb.run.alert(
                title="Training Complete",
                text=f"Training process on {config.dataset_name} has successfully finished.",
                level="INFO",
            )
    except Exception as e:
        if rank == 0:
            wandb.run.alert(
                title="Training Error",
                text=f"An error occurred during training on {config.dataset_name}: {e}",
                level="ERROR",
            )
        raise e


def test(
    config: DictConfig,
) -> None:
    world_size = torch.cuda.device_count()
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(rank)
    else:
        rank = 0

    if rank == 0:
        wandb.init(
            project=config.project_name,
            name=config.model_detail,
        )

    if "seed" in config:
        set_seed(config.seed)

    setup = SetUp(config)

    test_dataset = setup.get_test_dataset()
    sampler = (
        DistributedSampler(
            test_dataset,
            shuffle=False,
        )
        if world_size > 1
        else None
    )
    test_loader = build_test_dataloader(
        test_dataset=test_dataset,
        config=config,
        num_workers=setup.num_workers,
        sampler=sampler,
    )

    model = setup.get_model()
    data_encoder = setup.get_data_encoder()

    model.to(rank)

    try:
        results = generate_test_results(
            test_loader=test_loader,
            model=model,
            data_encoder=data_encoder,
            config=config,
            device=rank,
            tqdm_desc=f"Test {config.dataset_name}",
            tqdm_disable=(rank != 0),
        )

        if world_size > 1:
            dist.barrier()
            all_results = [None] * world_size
            dist.gather_object(
                results,
                all_results if rank == 0 else None,
                dst=0,
            )
            if rank == 0:
                results = [item for sublist in all_results for item in sublist]

        if rank == 0:
            df = save_test_results_json(
                results=results,
                output_dir=config.test_output_dir,
                output_name=config.test_output_name,
            )
            wandb.log({"test_results": wandb.Table(dataframe=df)})

            wandb.run.alert(
                title="Testing Complete",
                text=f"Testing process on {config.dataset_name} has successfully finished.",
                level="INFO",
            )
    except Exception as e:
        if rank == 0:
            wandb.run.alert(
                title="Testing Error",
                text=f"An error occurred during testing on {config.dataset_name}: {e}",
                level="ERROR",
            )
        raise e
    finally:
        if world_size > 1:
            dist.destroy_process_group()


def test_large(
    config: DictConfig,
) -> None:
    wandb.init(
        project=config.project_name,
        name=config.model_detail,
    )

    if "seed" in config:
        set_seed(config.seed)

    setup = SetUp(config)

    test_dataset = setup.get_test_dataset()
    test_loader = build_test_dataloader(
        test_dataset=test_dataset,
        config=config,
        num_workers=setup.num_workers,
        sampler=None,
    )

    model = setup.get_model()
    data_encoder = setup.get_data_encoder()

    try:
        results = generate_test_results(
            test_loader=test_loader,
            model=model,
            data_encoder=data_encoder,
            config=config,
            device=model.device,
            tqdm_desc=f"Test {config.dataset_name}",
            tqdm_disable=False,
        )

        df = save_test_results_json(
            results=results,
            output_dir=config.test_output_dir,
            output_name=config.test_output_name,
        )
        wandb.log({"test_results": wandb.Table(dataframe=df)})

        wandb.run.alert(
            title="Large Model Testing Complete",
            text=f"Testing process on {config.dataset_name} has successfully finished.",
            level="INFO",
        )
    except Exception as e:
        wandb.run.alert(
            title="Large Model Testing Error",
            text=f"An error occurred during testing on {config.dataset_name}: {e}",
            level="ERROR",
        )
        raise e


def test_vllm(
    config: DictConfig,
) -> None:
    wandb.init(
        project=config.project_name,
        name=config.model_detail,
    )

    if "seed" in config:
        set_seed(config.seed)

    setup = SetUp(config)

    data_encoder = setup.get_data_encoder()

    num_gpus = torch.cuda.device_count()
    tp_size = resolve_vllm_tp_size(
        config=config,
        num_gpus=num_gpus,
    )
    llm = build_vllm(
        config=config,
        tp_size=tp_size,
    )

    eos_token_id = (
        data_encoder.eos_token_id
        if config.modality == "text"
        else data_encoder.tokenizer.eos_token_id
    )

    sampling_params = build_sampling_params(
        config=config,
        stop_token_ids=[eos_token_id],
    )
    lora_request = build_lora_request(
        config=config,
        lora_int_id=1,
    )
    df = load_test_dataframe(
        config=config,
    )

    prompts = []
    labels = []

    if config.data_type == "conversational":
        for _, row in df.iterrows():
            conversation = row[config.conversation_column_name]
            preprocessed_conversation = [
                {
                    config.role_column_name: turn[config.role_column_name],
                    config.content_column_name: turn[config.content_column_name],
                }
                for turn in conversation
            ]
            label = preprocessed_conversation.pop()[config.content_column_name]

            prompt = data_encoder.apply_chat_template(
                conversation=preprocessed_conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=config.is_enable_thinking,
            )
            prompts.append(prompt)
            labels.append(label)

    elif config.data_type == "structural":
        for _, row in df.iterrows():
            data = row[config.data_column_name]
            label = row[config.target_column_name].strip()

            conversation = [
                {
                    config.role_column_name: "user",
                    config.content_column_name: data,
                },
            ]

            prompt = data_encoder.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=config.is_enable_thinking,
            )
            prompts.append(prompt)
            labels.append(label)

    try:
        outputs = llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            lora_request=lora_request if config.is_peft else None,
            use_tqdm=True,
        )

        results = []
        for output, label in zip(outputs, labels):
            instruction = output.prompt
            generation = output.outputs[0].text.strip()
            results.append(
                {
                    "instruction": instruction,
                    "generation": generation,
                    "label": label,
                }
            )

        os.makedirs(
            config.test_output_dir,
            exist_ok=True,
        )
        test_output_path = os.path.join(
            config.test_output_dir,
            f"{config.test_output_name}.json",
        )

        df = pd.DataFrame(results)
        df.to_json(
            test_output_path,
            orient="records",
            indent=2,
            force_ascii=False,
        )

        wandb.log({"test_results": wandb.Table(dataframe=df)})

        wandb.run.alert(
            title="vLLM Testing Complete",
            text=f"Testing process on {config.dataset_name} has successfully finished.",
            level="INFO",
        )
    except Exception as e:
        wandb.run.alert(
            title="vLLM Testing Error",
            text=f"An error occurred during testing on {config.dataset_name}: {e}",
            level="ERROR",
        )
        raise e


def test_vllm_multi_turn(
    config: DictConfig,
) -> None:
    wandb.init(
        project=config.project_name,
        name=config.model_detail,
    )

    if "seed" in config:
        set_seed(config.seed)

    setup = SetUp(config)

    data_encoder = setup.get_data_encoder()

    num_gpus = torch.cuda.device_count()
    tp_size = resolve_vllm_tp_size(
        config=config,
        num_gpus=num_gpus,
    )
    llm = build_vllm(
        config=config,
        tp_size=tp_size,
    )

    model_max_len = llm.llm_engine.model_config.max_model_len

    sampling_params = build_sampling_params(
        config=config,
        stop_token_ids=[data_encoder.eos_token_id],
    )
    lora_request = build_lora_request(
        config=config,
        lora_int_id=0,
    )
    df = load_test_dataframe(
        config=config,
    )

    try:
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating responses"):
            contents = row[config.content_column_name]

            if isinstance(contents, list):
                conversation = []
                generations = []

                for content in contents:
                    conversation.append(
                        {
                            config.role_column_name: "user",
                            config.content_column_name: content,
                        }
                    )
                    prompt = data_encoder.apply_chat_template(
                        conversation=conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=config.is_enable_thinking,
                    )

                    prompt_token_ids = data_encoder.encode(prompt)
                    if len(prompt_token_ids) >= model_max_len:
                        print(
                            f"Prompt length ({len(prompt_token_ids)}) is exceeding model max length ({model_max_len}). "
                            f"Skipping this turn."
                        )
                        generation = "MODEL_MAX_LENGTH_EXCEEDED"
                        generations.append(generation)
                        break

                    output = llm.generate(
                        prompts=prompt,
                        sampling_params=sampling_params,
                        lora_request=lora_request if config.is_peft else None,
                        use_tqdm=False,
                    )
                    generation = output[0].outputs[0].text.strip()
                    generations.append(generation)

                    conversation.append(
                        {
                            config.role_column_name: "assistant",
                            config.content_column_name: generation,
                        }
                    )

                result_item = row.to_dict()
                result_item["generation"] = generations
                results.append(result_item)
            else:
                conversation = [
                    {
                        config.role_column_name: "user",
                        config.content_column_name: contents,
                    }
                ]
                prompt = data_encoder.apply_chat_template(
                    conversation=conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=config.is_enable_thinking,
                )

                output = llm.generate(
                    prompts=prompt,
                    sampling_params=sampling_params,
                    lora_request=lora_request if config.is_peft else None,
                    use_tqdm=False,
                )
                generation = output[0].outputs[0].text.strip()

                result_item = row.to_dict()
                result_item["generation"] = generation
                results.append(result_item)

        os.makedirs(
            config.test_output_dir,
            exist_ok=True,
        )
        test_output_path = os.path.join(
            config.test_output_dir,
            f"{config.test_output_name}.jsonl",
        )

        result_df = pd.DataFrame(results)
        result_df.to_json(
            test_output_path,
            orient="records",
            lines=True,
            force_ascii=False,
        )

        for column in result_df.columns:
            result_df[column] = result_df[column].apply(
                lambda value: (
                    json.dumps(
                        value,
                        ensure_ascii=False,
                    )
                    if isinstance(value, (list, dict, set))
                    else value
                )
            )

        wandb.log({"test_results": wandb.Table(dataframe=result_df)})
        wandb.run.alert(
            title="vLLM Multi-Turn Testing Complete",
            text=f"Testing process on {config.dataset_name} has successfully finished.",
            level="INFO",
        )

    except Exception as e:
        wandb.run.alert(
            title="vLLM Multi-Turn Testing Error",
            text=f"An error occurred during testing on {config.dataset_name}: {e}",
            level="ERROR",
        )
        raise e
