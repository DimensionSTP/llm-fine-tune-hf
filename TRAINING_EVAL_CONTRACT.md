# Training and Evaluation Contract

This document defines the minimum execution contract for `llm-fine-tune-hf`.

## Runtime Inputs

- Required environment variables:
  - `PROJECT_DIR`
  - `CONNECTED_DIR`
  - `DEVICES`
  - `HF_HOME`
  - `USER_NAME`
- Primary config entrypoint: `configs/sft.yaml`.
- Main execution command:
  - `python main.py mode=<mode>`

## Supported Modes

- `train`
- `test`
- `test_large`
- `test_vllm`
- `test_vllm_multi_turn`

Any other mode must fail with `ValueError`.

## Output Path Contract

From `configs/sft.yaml`:

- Training checkpoints:
  - `${connected_dir}/checkpoints/${model_name}/${dataset_name}/${strategy}/${save_detail}`
- Test outputs:
  - `${connected_dir}/tests/${model_type}`

When mode is `train`, checkpoints must be written to `output_dir`.
When mode is a test mode, result artifacts must be written under `test_output_dir`.

## Backward-Compatibility Rule

- Changes to mode names, required env vars, or default output path keys must be reflected in:
  - `README.md`
  - `CHANGELOG.md`
  - this contract document
