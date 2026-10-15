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
- `test_decision`
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

Training option contracts:

- Decision reuses the causal LM/VLM and ordinary adapters with no new head or vocabulary. Structural and conversational datasets have separate classes/config targets; training and test profiles are separate. Two candidates use score-difference BCE, larger sets use masked CE, and gradient accumulation counts questions rather than tokens/candidates. Hard and soft targets remain distinct; accuracy/ECE exclude soft-only labels. Dynamic/fixed and left/right padding score the final real token, preserve image tensors, and reject overlong inputs without truncation. Same-run resume requires unchanged decision configuration, marker mapping, and template; cross-method training uses continuation in a new run.
- Hub kernels are training-only opt-in through `model_loading.train.use_kernels`; a configured Hydra `transformers.KernelConfig` requires that flag, and the selected TrainingArguments must disable Liger. Custom mappings retain inference entries and gain separate training entries. The pinned upstream custom loader needs explicit accelerator placement; default mappings keep the existing model-loading placement.
- SFT keeps `sft_label_mask.backend=repo` and reentrant checkpointing by default. Native masks require a fast selected encoder with generation-tag templates and preserve VLM image inputs; selective checkpointing requires non-reentrant mode and excludes PEFT with ZeRO-3.
- Distillation tool calling reuses `agentic` with dataset prompts and synchronous tools only. Environment factories and rollout workers remain GRPO/Async GRPO settings.
- RLOO uses the stable upstream trainer with the existing GRPO dataset and reward manager. Text and VLM inputs share the configured image controls. VLM requires conversational prompts and decoded images; image-path datasets use `decode_image_paths=True`. RLOO defaults to left padding so multimodal token types and left-padded prompt tokens remain aligned in unequal-length batches. Its generation cadence follows gradient accumulation by default, generation groups require at least two completions, and generation/global evaluation batches must be divisible by their generation counts. GRPO-only streaming, agentic, loss, and completion-termination options are not forwarded.
- Enabled GRPO/RLOO/Distillation completion tables use the selected backend and retain the upstream text or structured messages with configured-secret redaction. GRPO and RLOO require their reward, extra, and advantage buffers; Distillation has only prompt/completion buffers. SDPO is not attached to this callback. Completion logging stays disabled by default.

Dependency notes:

- Python 3.11 or newer is required; the validated interpreter is Python 3.12.
- Use the documented CUDA 12.9 install path, the FlashInfer CUDA index for architecture-specific JIT providers, and `requirements-overrides.txt` for the explicit Transformers metadata exception; this does not imply a clean `pip check`. MLflow supports the pinned Pandas version without an override.
- On Blackwell with a driver that cannot execute Triton's bundled CUDA 13.1 PTX, the existing `.env` bootstrap may select `TRITON_PTXAS_BLACKWELL_PATH=${CONDA_PREFIX}/bin/ptxas` from the active CUDA 12.9 environment. Child processes inherit that compiler selection.
- Torch changes require rebuilding Flash Attention and causal-conv1d against the active Torch ABI; Conda CUDA builds may require the toolkit header path described in README.
- Base install excludes `flash-attn`.
- Optional GPU install path:
  - `uv pip install --override requirements-overrides.txt --torch-backend=cu129 --extra-index-url https://flashinfer.ai/whl/cu129 --no-build-isolation ".[gpu]"`
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
- `test_decision`: use `${connected_dir}/tests/${model_detail}/decision/${dataset_name}.json` and `${dataset_name}_manifest.json` by default. Return candidate probabilities and typed choice/Noul/Score values without generation; the manifest records `generation=null`. Test targets are optional, but supplied invalid targets must fail validation. Rank-padding duplicates are removed by dataset row index, not by user ID; intentional resampling repetitions remain distinct.
- `test*`: companion manifests must store the full resolved config once and add the active data encoder, resolved test files, runtime backend, effective device map or tensor-parallel size, and actual generation parameters; sensitive credential values anywhere in the companion must be represented as `<redacted>`.
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
