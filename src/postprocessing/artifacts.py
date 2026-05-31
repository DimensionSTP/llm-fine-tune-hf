import os

from omegaconf import DictConfig


def resolve_existing_artifact_output_dir(
    config: DictConfig,
) -> None:
    if config.output_dir is not None:
        return

    if config.run_id is None:
        raise ValueError("run_id is required to resolve an existing artifact path.")

    config.output_dir = os.path.join(
        config.output_base_dir,
        config.run_id,
    )
