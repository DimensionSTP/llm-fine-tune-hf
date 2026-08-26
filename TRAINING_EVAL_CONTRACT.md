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
- `SLACK_WEBHOOK_URL` when the default `notifications=slack` backend is used

Credentials must be supplied through `.env` or environment variables, not Hydra CLI overrides.

Dependency notes:

- Base install excludes `flash-attn`.
- Optional GPU install path:
  - `uv pip install --override requirements-overrides.txt --torch-backend=cu129 --no-build-isolation ".[gpu]"`
  - or pinned Git install command from README.

## Tracking Contract

- The portable default is `tracking=mlflow`, which uses local SQLite and file artifacts.
- `tracking=wandb` uses W&B.
- `tracking=mlflow_server` requires `MLFLOW_TRACKING_URI`, uses the optional MLflow basic-auth environment variables, and does not set a client-side experiment artifact location.
- Only rank 0 owns a tracking run during distributed execution.
- MLflow normal completion, ordinary exceptions, and `KeyboardInterrupt` or `SystemExit` terminate as `FINISHED`, `FAILED`, and `KILLED`, respectively.
- Tracking alert or finalization failures during exception handling must not replace the original pipeline exception.
- Forced process death such as `SIGKILL` cannot be finalized by Python and may leave a stale tracking run.
- Failures during dataset, model, encoder, or Trainer setup after tracking initialization must not leave an active run.
- Resume reuses the existing `tracking_run_id` from `${output_dir}/tracking_metadata.json`.
- MLflow validates every train metadata artifact before upload and fails closed without uploading any metadata when an unredacted secret remains.
- Notification and tracking preserving-error logs must redact repository-known secrets without changing the runtime credential value or replacing the original pipeline exception.

## Output Contract

- `train`: checkpoint/model artifacts must be written to config-defined output directory:
  `${connected_dir}/checkpoints/${model_name}/${effective_dataset_name}/${strategy}/${save_detail}/${run_id}`.
- `train`: `run_id` must be allocated automatically by Python as an ordered `run-000N` leaf.
- `train`: runtime batch-size fields must be logged as metadata, not embedded in `save_detail`.
- `train`: `run_manifest.json` and `resolved_config.yaml` must exist before preflight, tracking, or setup. The manifest must record `prepared`, `running`, `completed`, `failed`, or `interrupted` and the last `preflight`, `setup`, `training`, `saving`, or `completed` stage.
- `train`: `resolved_config.yaml` is the complete config source of truth with sensitive credential values stored as `<redacted>` and non-sensitive values fully resolved; `training_args.json` records instantiated trainer arguments, and `run_manifest.json` records observed inputs, PEFT lineage, runtime facts, failures, and relative internal artifact references without duplicating either file.
- `train`: Hydra must not persist a duplicate `.hydra` metadata directory because the repository-owned metadata files are canonical.
- MLflow `train`: upload available lifecycle metadata under the run's `metadata/` artifact path before finalizing `FINISHED`, `FAILED`, or `KILLED`; completed runs must preserve `run_manifest.json`, `resolved_config.yaml`, `training_args.json`, and `tracking_metadata.json`, while failed or interrupted runs preserve every file created before termination.
- `test`, `test_large`, and `test_vllm`: the canonical result and companion manifest must be written under `${connected_dir}/tests/${model_detail}` as `${dataset_name}.json` and `${dataset_name}_manifest.json`.
- `test_vllm_multi_turn`: the result and companion manifest must be written under the same model-detail namespace as `${dataset_name}_multi_turn.jsonl` and `${dataset_name}_multi_turn_manifest.json`.
- `test*`: companion manifests must store the full resolved config once and add the active data encoder, resolved test files, runtime backend, effective device map or tensor-parallel size, and actual generation parameters.
- PEFT `test_vllm` and `test_vllm_multi_turn`: `peft_test.adapter_path` is the immutable source adapter, and adapter validation and configured vLLM parameter-name remapping must complete before vLLM model loading.
- PEFT `test_vllm` and `test_vllm_multi_turn`: passthrough must use the source adapter without creating an artifact; non-passthrough must use a validated content-addressed adapter under `${test_output_dir}/vllm_lora_adapters/<profile>-<identity>/` without modifying the source.
- PEFT `test_vllm` and `test_vllm_multi_turn`: the companion manifest must keep source weights/config hashes under `resolved_input.peft_adapter` and effective path/hash, tensor counts, and materialization action under `runtime.vllm_lora_adapter`; remap policy and source path remain in the resolved config and must not be duplicated in runtime metadata.
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
