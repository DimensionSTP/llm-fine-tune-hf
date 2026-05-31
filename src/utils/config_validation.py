import os

from omegaconf import DictConfig


def validate_training_arguments_config(
    config: DictConfig,
) -> None:
    if config.fine_tune_method != "sft":
        return
    if (
        config.sft_loss_type == "chunked_nll"
        and config.training_arguments.use_liger_kernel
    ):
        raise ValueError(
            "SFT loss_type='chunked_nll' is not compatible with "
            "use_liger_kernel=True."
        )


def validate_train_artifact_config(
    config: DictConfig,
) -> None:
    if config.mode != "train":
        return

    if config.run_id is None:
        raise ValueError("run_id must be allocated before train starts.")

    if config.output_dir is None:
        raise ValueError("output_dir must be allocated before train starts.")

    output_dir = str(config.output_dir)
    if not os.path.exists(output_dir):
        raise ValueError(f"output_dir does not exist: {output_dir}")

    if not config.resume_training:
        return

    if config.resume_from_checkpoint is None:
        raise ValueError(
            "resume_from_checkpoint is required when resume_training is true."
        )

    resume_from_checkpoint = str(config.resume_from_checkpoint)
    if not os.path.exists(resume_from_checkpoint):
        raise ValueError(
            f"resume_from_checkpoint does not exist: {resume_from_checkpoint}"
        )
