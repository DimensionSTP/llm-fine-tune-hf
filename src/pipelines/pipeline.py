import os

from hydra.utils import get_class
from omegaconf import DictConfig, ListConfig, OmegaConf

import json

import pandas as pd

import torch
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler

from transformers import set_seed

from tqdm import tqdm

from ..helpers import build_enable_thinking_kwargs
from ..datasets import (
    build_vllm_prompt_payload,
    collect_vllm_images,
    is_vlm_content_parts,
)
from ..utils import *


def train(
    config: DictConfig,
) -> None:
    rank = int(os.environ.get("RANK", 0))
    validate_peft_initialization_config(config=config)
    run_memory_preflight_if_needed(
        config=config,
        rank=rank,
    )
    prepare_train_artifact_config(
        config=config,
        rank=rank,
    )
    if rank == 0:
        validate_train_artifact_config(
            config=config,
        )
        init_train_tracking(config=config)

    if "seed" in config:
        set_seed(config.seed)

    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    async_runtime_state = resolve_async_runtime_state(
        config=config,
        rank=rank,
    )
    async_runtime_enabled = async_runtime_state["enabled"]

    if (
        (not is_distributed)
        and (config.devices is not None)
        and (not async_runtime_enabled)
    ):
        if isinstance(config.devices, int):
            num_gpus = min(config.devices, torch.cuda.device_count())
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))
        elif isinstance(config.devices, str):
            os.environ["CUDA_VISIBLE_DEVICES"] = config.devices
        elif isinstance(config.devices, (list, ListConfig)):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.devices))

    if rank == 0:
        validate_distributed_runtime_config(
            config=config,
            runtime_snapshot=None,
        )

    setup = SetUp(config)
    if run_async_inference_server(
        config=config,
        runtime_state=async_runtime_state,
    ):
        return

    async_vllm_process, async_vllm_log_handle = start_async_training_runtime(
        config=config,
        runtime_state=async_runtime_state,
    )

    train_datasets = setup.get_train_datasets()
    train_dataset = train_datasets["train"]
    val_dataset = train_datasets["val"]
    write_memory_preflight_selection(
        config=config,
        train_dataset=train_dataset,
    )
    train_dataset = apply_memory_preflight_dataset(
        config=config,
        train_dataset=train_dataset,
    )

    ds_config = setup.get_ds_config()
    training_arguments = setup.get_training_arguments(
        ds_config=ds_config,
    )

    data_encoder = setup.get_data_encoder()
    data_collator = setup.get_data_collator(data_encoder=data_encoder)

    training_arguments = setup.finalize_training_arguments(
        training_arguments=training_arguments,
        data_encoder=data_encoder,
    )
    write_run_metadata(
        config=config,
        training_arguments=training_arguments,
        rank=rank,
    )

    if config.fine_tune_method == "async_grpo":
        model = config.pretrained_model_name
        if config.is_preprocessed:
            merged_model_path = os.path.join(
                config.merged_model_path,
                config.pretrained_model_name,
            )
            if os.path.exists(merged_model_path):
                model = merged_model_path
    else:
        model = setup.get_model()

    trainer_config = OmegaConf.to_container(
        config.trainer,
        resolve=True,
    )
    trainer_config.pop(
        "_target_",
        None,
    )

    TrainerClass = get_class(config.trainer._target_)

    reward_manager = None
    if config.fine_tune_method in {"grpo", "async_grpo", "sdpo", "a2po"}:
        reward_manager = setup.get_reward_manager()
        trainer_config["reward_funcs"] = reward_manager.get_reward_funcs()

    if config.fine_tune_method in {"gkd", "gold"}:
        trainer_config["teacher_model"] = config.teacher.model

    trainer_kwargs = {
        "model": model,
        "args": training_arguments,
        "train_dataset": train_dataset,
        "processing_class": data_encoder,
        **trainer_config,
    }
    if data_collator is not None:
        trainer_kwargs["data_collator"] = data_collator
    if config.fine_tune_method != "async_grpo":
        trainer_kwargs["eval_dataset"] = val_dataset

    try:
        trainer = TrainerClass(
            **trainer_kwargs,
        )
        if patch_qwen_packed_moe_vllm_sync(
            trainer=trainer,
            config=config,
        ):
            print(
                "[patch] Applied Qwen packed-MoE vLLM sync filter for router-with-lora GRPO."
            )
        elif patch_sparse_decoder_moe_vllm_sync(
            trainer=trainer,
            config=config,
        ):
            print(
                "[patch] Applied sparse-decoder MoE vLLM sync filter for router-with-lora GRPO."
            )
        elif patch_lora_streaming_vllm_sync(
            trainer=trainer,
            config=config,
        ):
            print("[patch] Applied streaming LoRA vLLM sync for GRPO.")
        if patch_grpo_completion_termination(
            trainer=trainer,
            config=config,
        ):
            print("[patch] Applied GRPO completion termination override.")
        if (
            config.fine_tune_method in {"grpo", "sdpo"}
            and reward_manager is not None
            and config.use_vllm
            and config.vllm_mode == "colocate"
            and hasattr(trainer, "vllm_generation")
        ):
            prepare_colocated_vllm_models(
                reward_manager=reward_manager,
                generation_model=trainer.vllm_generation.llm,
            )

        trainer.train(
            resume_from_checkpoint=(
                config.resume_from_checkpoint if config.resume_training else None
            )
        )
        trainer.save_model()

        if rank == 0:
            alert_tracking(
                config=config,
                title="Training Complete",
                text=f"Training process on {config.dataset_name} has successfully finished.",
                level="INFO",
            )
    except Exception as e:
        if rank == 0:
            alert_tracking(
                config=config,
                title="Training Error",
                text=f"An error occurred during training on {config.dataset_name}: {e}",
                level="ERROR",
            )
        raise e
    finally:
        stop_async_training_runtime(
            config=config,
            runtime_state=async_runtime_state,
            process=async_vllm_process,
            log_handle=async_vllm_log_handle,
        )
        if rank == 0:
            finish_tracking(config=config)


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
        init_eval_tracking(config=config)

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
        dataloader_kwargs=setup.get_dataloader_kwargs(),
        sampler=sampler,
    )

    data_encoder = setup.get_data_encoder()
    model = setup.get_model()

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
            log_tracking_table(
                config=config,
                key="test_results",
                dataframe=df,
            )

            alert_tracking(
                config=config,
                title="Testing Complete",
                text=f"Testing process on {config.dataset_name} has successfully finished.",
                level="INFO",
            )
    except Exception as e:
        if rank == 0:
            alert_tracking(
                config=config,
                title="Testing Error",
                text=f"An error occurred during testing on {config.dataset_name}: {e}",
                level="ERROR",
            )
        raise e
    finally:
        if rank == 0:
            finish_tracking(config=config)
        if world_size > 1:
            dist.destroy_process_group()


def test_large(
    config: DictConfig,
) -> None:
    init_eval_tracking(config=config)

    if "seed" in config:
        set_seed(config.seed)

    setup = SetUp(config)

    test_dataset = setup.get_test_dataset()
    test_loader = build_test_dataloader(
        test_dataset=test_dataset,
        config=config,
        dataloader_kwargs=setup.get_dataloader_kwargs(),
        sampler=None,
    )

    data_encoder = setup.get_data_encoder()
    model = setup.get_model()

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
        log_tracking_table(
            config=config,
            key="test_results",
            dataframe=df,
        )

        alert_tracking(
            config=config,
            title="Large Model Testing Complete",
            text=f"Testing process on {config.dataset_name} has successfully finished.",
            level="INFO",
        )
    except Exception as e:
        alert_tracking(
            config=config,
            title="Large Model Testing Error",
            text=f"An error occurred during testing on {config.dataset_name}: {e}",
            level="ERROR",
        )
        raise e
    finally:
        finish_tracking(config=config)


def test_vllm(
    config: DictConfig,
) -> None:
    init_eval_tracking(config=config)

    if "seed" in config:
        set_seed(config.seed)

    setup = SetUp(config)

    data_encoder = setup.get_data_encoder()
    df = load_test_dataframe(
        config=config,
    )

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
            chat_template_kwargs = build_enable_thinking_kwargs(
                data_encoder=data_encoder,
                is_enable_thinking=config.is_enable_thinking,
            )

            prompt = data_encoder.apply_chat_template(
                conversation=preprocessed_conversation,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
            images = (
                collect_vllm_images(
                    value=preprocessed_conversation,
                    dataset_image=config.dataset_image,
                    data_path=config.data_path,
                )
                if config.modality != "text"
                else []
            )
            prompts.append(
                build_vllm_prompt_payload(
                    prompt=prompt,
                    images=images,
                )
            )
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
            chat_template_kwargs = build_enable_thinking_kwargs(
                data_encoder=data_encoder,
                is_enable_thinking=config.is_enable_thinking,
            )

            prompt = data_encoder.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
            images = (
                collect_vllm_images(
                    value=data,
                    dataset_image=config.dataset_image,
                    data_path=config.data_path,
                )
                if config.modality != "text"
                else []
            )
            prompts.append(
                build_vllm_prompt_payload(
                    prompt=prompt,
                    images=images,
                )
            )
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

        log_tracking_table(
            config=config,
            key="test_results",
            dataframe=df,
        )

        alert_tracking(
            config=config,
            title="vLLM Testing Complete",
            text=f"Testing process on {config.dataset_name} has successfully finished.",
            level="INFO",
        )
    except Exception as e:
        alert_tracking(
            config=config,
            title="vLLM Testing Error",
            text=f"An error occurred during testing on {config.dataset_name}: {e}",
            level="ERROR",
        )
        raise e
    finally:
        finish_tracking(config=config)


def test_vllm_multi_turn(
    config: DictConfig,
) -> None:
    init_eval_tracking(config=config)

    if "seed" in config:
        set_seed(config.seed)

    setup = SetUp(config)

    data_encoder = setup.get_data_encoder()
    text_encoder = resolve_text_encoder(data_encoder=data_encoder)
    df = load_test_dataframe(
        config=config,
    )

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
        stop_token_ids=[text_encoder.eos_token_id],
    )
    lora_request = build_lora_request(
        config=config,
        lora_int_id=0,
    )

    try:
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating responses"):
            contents = row[config.content_column_name]

            if isinstance(contents, list) and not is_vlm_content_parts(value=contents):
                conversation = []
                generations = []

                for content in contents:
                    conversation.append(
                        {
                            config.role_column_name: "user",
                            config.content_column_name: content,
                        }
                    )
                    chat_template_kwargs = build_enable_thinking_kwargs(
                        data_encoder=data_encoder,
                        is_enable_thinking=config.is_enable_thinking,
                    )
                    prompt = data_encoder.apply_chat_template(
                        conversation=conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                        **chat_template_kwargs,
                    )

                    prompt_token_ids = text_encoder.encode(prompt)
                    if len(prompt_token_ids) >= model_max_len:
                        print(
                            f"Prompt length ({len(prompt_token_ids)}) is exceeding model max length ({model_max_len}). "
                            f"Skipping this turn."
                        )
                        generation = "MODEL_MAX_LENGTH_EXCEEDED"
                        generations.append(generation)
                        break

                    images = (
                        collect_vllm_images(
                            value=conversation,
                            dataset_image=config.dataset_image,
                            data_path=config.data_path,
                        )
                        if config.modality != "text"
                        else []
                    )
                    prompt_payload = build_vllm_prompt_payload(
                        prompt=prompt,
                        images=images,
                    )
                    output = llm.generate(
                        prompts=prompt_payload,
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
                chat_template_kwargs = build_enable_thinking_kwargs(
                    data_encoder=data_encoder,
                    is_enable_thinking=config.is_enable_thinking,
                )
                prompt = data_encoder.apply_chat_template(
                    conversation=conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                    **chat_template_kwargs,
                )

                images = (
                    collect_vllm_images(
                        value=conversation,
                        dataset_image=config.dataset_image,
                        data_path=config.data_path,
                    )
                    if config.modality != "text"
                    else []
                )
                prompt_payload = build_vllm_prompt_payload(
                    prompt=prompt,
                    images=images,
                )
                output = llm.generate(
                    prompts=prompt_payload,
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

        log_tracking_table(
            config=config,
            key="test_results",
            dataframe=result_df,
        )
        alert_tracking(
            config=config,
            title="vLLM Multi-Turn Testing Complete",
            text=f"Testing process on {config.dataset_name} has successfully finished.",
            level="INFO",
        )

    except Exception as e:
        alert_tracking(
            config=config,
            title="vLLM Multi-Turn Testing Error",
            text=f"An error occurred during testing on {config.dataset_name}: {e}",
            level="ERROR",
        )
        raise e
    finally:
        finish_tracking(config=config)
