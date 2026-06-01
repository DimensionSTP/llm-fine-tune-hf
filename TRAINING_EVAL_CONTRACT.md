# Training and Evaluation Contract

## Scope

Contract for `llm-fine-tune-hf` runtime interface, supported execution modes, and expected outputs.

## Entry Point

- Main command: `python main.py mode=<mode>`
- Config root: `configs/`
- Default config: `configs/sft.yaml`

## Supported Modes

- `train`
- `test`
- `test_large`
- `test_vllm`
- `test_vllm_multi_turn`

Unsupported mode must fail with `ValueError`.

## Required Runtime Inputs

Environment variables:

- `PROJECT_DIR`
- `CONNECTED_DIR`
- `DEVICES`
- `HF_HOME`
- `USER_NAME`

Dependency notes:

- Base install excludes `flash-attn`.
- Optional GPU install path:
  - `pip install ".[gpu]"`
  - or pinned Git install command from README.

## Output Contract

- `train`: checkpoint/model artifacts must be written to config-defined output directory:
  `${connected_dir}/checkpoints/${model_name}/${dataset_name}/${strategy}/${save_detail}/${run_id}`.
- `train`: `run_id` must be allocated automatically by Python as an ordered `run-000N` leaf.
- `train`: runtime batch-size fields must be logged as metadata, not embedded in `save_detail`.
- `train`: `run_manifest.json` must record planned and observed distributed, device, and batch runtime metadata for single-node and multi-node runs.
- `test*`: evaluation/generation artifacts must be written to mode-specific test output paths.
- Runs must log enough metadata (model, dataset, key runtime options) for reproducibility.
- Postprocessing scripts must keep `run_id` as a script-local variable, let the Python entrypoint resolve artifact paths from config-composed `output_base_dir`, and must not require command-line or environment overrides at execution time.

## Compatibility Rules

When changing mode names, required env vars, or output path schema, update in same change-set:

- `README.md`
- `USAGE_GUIDE.md`
- this contract
- `CHANGELOG.md`

## Validation Checklist

1. All supported modes start successfully with valid config.
2. Invalid mode fails explicitly.
3. Output artifacts are generated for each tested mode.
4. Optional `flash-attn` path is documented and tested in GPU environment.
