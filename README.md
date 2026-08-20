# LLM model fine-tuning & inference pipeline

## For (s)LLM model fine-tuning & inference

### Dataset

Any Structured Datasets

### Quick setup (conda)

```bash
# clone project
git clone https://github.com/DimensionSTP/llm-fine-tune-hf.git
cd llm-fine-tune-hf

# [OPTIONAL] create conda environment
conda create -n myenv python=3.12 -y
conda activate myenv

# install build requirements
python -m pip install uv==0.10.12
uv pip install "setuptools>=68" wheel

# install requirements
uv pip install \
    --override requirements-overrides.txt \
    --torch-backend=cu129 \
    --no-build-isolation \
    -r requirements.txt
```

`requirements-overrides.txt` keeps Pandas 3.0.1 while overriding MLflow 3.13.0's `pandas<3` package metadata.

### Quick setup (pyproject.toml)

```bash
# install build requirements
python -m pip install uv==0.10.12
uv pip install "setuptools>=68" wheel

# install project dependencies from pyproject.toml
uv pip install \
    --override requirements-overrides.txt \
    --torch-backend=cu129 \
    --no-build-isolation \
    .

# [OPTIONAL] editable install for development
uv pip install \
    --override requirements-overrides.txt \
    --torch-backend=cu129 \
    --no-build-isolation \
    -e .
```

### Optional GPU dependency (flash-attn)

```bash
# Option A: install optional GPU extra from pyproject
uv pip install \
    --override requirements-overrides.txt \
    --torch-backend=cu129 \
    --no-build-isolation \
    ".[gpu]"

# Option B: install directly from pinned Git commit
python -m pip install --no-build-isolation "flash-attn @ git+https://github.com/Dao-AILab/flash-attention.git@060c9188beec3a8b62b33a3bfa6d5d2d44975fab"
```

### Execution contract

- See `TRAINING_EVAL_CONTRACT.md` for required runtime inputs, supported modes, and output path expectations.

### .env file setting

```shell
PROJECT_DIR={PROJECT_DIR}
CONNECTED_DIR={CONNECTED_DIR}
DEVICES={DEVICES}
HF_HOME={HF_HOME}
USER_NAME={USER_NAME}
```

### Train

* end-to-end

```shell
python main.py mode=train
```

Training automatically allocates `run_id` values such as `run-0001` under the method/model/data checkpoint path. It atomically writes `run_manifest.json` and the full `resolved_config.yaml` before preflight, tracking initialization, or model setup, then adds `training_args.json` after the trainer arguments are instantiated. The manifest moves through `prepared`, `running`, and `completed`; ordinary exceptions become `failed`, while `KeyboardInterrupt` and `SystemExit` become `interrupted`. Its stage identifies `preflight`, `setup`, `training`, `saving`, or `completed`, so an incomplete run retains the last reached phase. Config-owned values, including every reward setting, remain in `resolved_config.yaml`; `training_args.json` records the effective trainer arguments; and `run_manifest.json` records observed input, PEFT, distributed, device, batch, DataLoader, lifecycle, and relative artifact facts without duplicating the config. Run-owned artifact references are relative to the manifest directory so a copied run remains internally navigable; external model, adapter, and dataset paths remain historical lineage in the config or resolved input. Runtime batch-size fields stay out of the checkpoint path. For distributed or multi-node runs, set `distributed.enabled=true` and configure `distributed.num_machines`, `distributed.num_processes_per_machine`, `distributed.machine_rank`, `distributed.main_process_ip`, and `distributed.main_process_port`. `run_metadata.allocation_timeout_seconds`, `run_metadata.allocation_poll_interval_seconds`, and `run_metadata.allocation_freshness_grace_seconds` control how non-rank0 processes wait for rank0's shared run directory allocation.

MLflow is the default experiment tracking backend, and Slack is the default notification backend through `SLACK_WEBHOOK_URL`. Use `tracking=wandb notifications=wandb` for W&B tracking and native alerts, `notifications=disabled` to disable lifecycle notifications, or `tracking=mlflow_server` for a remote MLflow server selected through `MLFLOW_TRACKING_URI`. Train runs persist only the selected backend and `tracking_run_id` in `${output_dir}/tracking_metadata.json`; checkpoint `run_id` remains local to `output_base_dir`, and interrupted-run resume reuses the persisted tracking identity instead of falling back to `run_id` or starting a new backend run. MLflow also records artifact `run_id` as a searchable `artifact_run_id` tag. Normal completion, ordinary exceptions, and `KeyboardInterrupt` or `SystemExit` end MLflow runs as `FINISHED`, `FAILED`, and `KILLED`, respectively.

### Test

* end-to-end

```shell
python main.py mode=test
```

* end-to-end(big model)

```shell
python main.py mode=test_large
```

* end-to-end(vLLM)

```shell
python main.py mode=test_vllm
```

* end-to-end(vLLM with multi-turn)

```shell
python main.py mode=test_vllm_multi_turn
```

`test_vllm` and `test_vllm_multi_turn` support VLM inputs by sending resolved images through vLLM `multi_modal_data` when `modality` is not `text`.

Test artifacts use the config-composed `${connected_dir}/tests/${model_detail}` directory. `test`, `test_large`, and `test_vllm` update the canonical `${dataset_name}.json` result and `${dataset_name}_manifest.json` companion. `test_vllm_multi_turn` writes `${dataset_name}_multi_turn.jsonl` and `${dataset_name}_multi_turn_manifest.json`. Each companion stores the full resolved config once, plus the active data encoder, resolved test files, runtime backend, effective device map or tensor-parallel size, and actual generation parameters.

### Examples of shell scipts

* full preprocessing

```shell
bash scripts/preprocessing/preprocess.sh
```

* dataset preprocessing

```shell
bash scripts/preprocessing/preprocess_dataset.sh
```

* train

```shell
bash scripts/train/train.sh
```

* multi-node train

```shell
# Edit the variables at the top of both rank scripts before running:
# - main_process_ip: replace RANK0_NODE_IP with the rank 0 node address
# - machine_rank: 0 on rank 0, 1 on rank 1
# - gpu_ids: visible GPU ids on each node
# - model_type defaults to Qwen3.5-9B
#
# Run the matching rank scripts on separate nodes.
# On rank 1 node:
bash scripts/train/multinode/sft_train_rank1.sh

# On rank 0 node:
bash scripts/train/multinode/sft_train_rank0.sh
```

```shell
# GRPO external server mode uses one trainer node and one vLLM server node.
# Edit VLLM_SERVER_NODE_IP in the trainer script and vllm_model/server settings
# in the vLLM script before running.
# Dense-model vLLM server mode keeps data_parallel_size=1. The server script
# derives tensor_parallel_size from gpu_ids so all visible server GPUs are used
# through tensor parallelism.
# On vLLM server node:
bash scripts/train/multinode/grpo_server_vllm_rank1.sh

# On trainer node:
bash scripts/train/multinode/grpo_server_train_rank0.sh
```

* async GRPO train

```shell
# repo async runtime starts/stops vLLM server automatically
# requirement: even number of GPUs and >=2 GPUs
# async_runtime.vllm_server.tensor_parallel_size=auto maps to the GPU count
# assigned to the vLLM server side.
bash scripts/train/async_grpo_train.sh
```

```shell
# config-only path (without script)
# world_size=1: half GPUs for vLLM and half GPUs for trainer
# world_size=2: rank0=trainer, rank1=vLLM server
python main.py --config-name=async_grpo.yaml mode=train
```

```shell
# safe stop
# script/main runtime stops managed server automatically on exit
```

* stable distillation train

```shell
bash scripts/train/distillation_train.sh
```

Stable Distillation loads the teacher locally and uses vLLM for student completion generation. The default is colocate mode with tensor parallel size 1; `vllm_mode=server` changes only the student generation backend, not the teacher location.

* test

```shell
bash scripts/test/test.sh
```

* test_large

```shell
bash scripts/test/test_large.sh
```

* test_vllm

```shell
bash scripts/test/test_vllm.sh
```

* test_vllm_multi_turn

```shell
bash scripts/test/test_vllm_multi_turn.sh
```

### Additional Options

* SFT train(masking input)

```shell
is_sft={True or False}
```

* SFT loss type

```shell
sft_loss_type={nll or chunked_nll}
```

`nll` is the default SFT loss type and is the supported default for assistant-only SFT (`is_sft=True`).

`chunked_nll` is an optional SFT-only loss path that reduces peak VRAM for long-context SFT while keeping the NLL objective. Use it for non-assistant-only long-context SFT when VRAM pressure is the bottleneck. Disable the default Liger kernel with `training_arguments.use_liger_kernel=False` before selecting `chunked_nll`, and do not use it with assistant-only SFT (`is_sft=True`). Smoke tests on both LLM and VLM paths showed that `chunked_nll + is_sft=True` can stall at the first training step. The likely reason is that `chunked_nll` drops `labels == -100` tokens before the LM head, while assistant-only masking makes valid label positions sparse and non-contiguous, causing the hidden-state gather/compaction path to become the bottleneck. `dynamic` padding is independent of the loss choice and is compatible with both `nll` and `chunked_nll`. The unsupported combinations are `chunked_nll + is_sft=True` and `chunked_nll + Liger`.

* SFT padding strategy

```shell
sft_padding_strategy={max_length or dynamic}
```

`dynamic` is the default. It keeps `max_length` as the truncation cap, pads each batch to the longest sample, uses `pad_to_multiple_of` when set, and is supported for both LLM and VLM SFT. `max_length` selects sample-level fixed padding. SFT dynamic padding is right-padding only; `left_padding=True` fails fast.

* Memory preflight

```shell
memory_preflight.enabled={true or false}
```

`false` is the default and preserves the existing training path. When enabled, memory preflight runs a strict subprocess probe before the real training run, selects a max-shape batch, and starts the real run only if the probe succeeds. The probe does not allocate a second run or initialize experiment tracking; its `command.json`, `selected_indices.json`, and `result.json` evidence is stored under `${output_dir}/memory_preflight` and referenced relatively by the parent manifest. The current implementation supports direct single-process and single-node multi-GPU SFT, DPO, KTO, GKD, GOLD, GRPO, SDPO, and A2PO preflight. vLLM preflight is limited to colocate mode; multi-node, vLLM server, and async GRPO topologies fail fast instead of running an approximate probe.

* DataLoader runtime policy

```shell
dataloader_runtime.mode={auto or manual}
```

`auto` is the default. It resolves DataLoader workers per process from node-local CPU capacity and local rank count, then records the resolved runtime in the run manifest while the configured policy remains in `resolved_config.yaml`. The same resolved policy is used by Trainer-backed training and standalone test DataLoaders. All workloads use `prefetch_factor=2` when workers are enabled. `persistent_workers` is enabled only when the resolved worker count is greater than zero.

* Tracking backend

```shell
tracking={wandb or mlflow or mlflow_server}
```

`mlflow` is the default. It uses `sqlite:///${connected_dir}/mlflow.db` and `file://${connected_dir}/mlflow-artifacts`, records system metrics, and writes `tracking_metadata.json` under each train `output_dir` so checkpoint artifact `run_id` can map to the MLflow run UUID on resume. GPU system metrics follow `CUDA_VISIBLE_DEVICES` instead of collecting unrelated physical GPUs on the same node. MLflow line charts provide exponential moving average smoothing from 0 to 100 without storing duplicate smoothed metrics. `mlflow_server` requires `MLFLOW_TRACKING_URI` and leaves experiment artifact location selection to the server; set `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` when the endpoint uses MLflow basic authentication. Resume requires existing tracking metadata for both tracking backends.

```shell
notifications={slack or disabled or wandb}
```

`slack` is the default and requires `SLACK_WEBHOOK_URL`. Slack notifications work with either tracking backend. `notifications=wandb` requires `tracking=wandb` and uses the active W&B run.

* Supported training methods

```shell
python main.py --config-name={method}.yaml mode=train
```

| Method | Config | Dataset contract | Notes |
| --- | --- | --- | --- |
| SFT | `sft.yaml` | SFT | `nll` is the default loss type; `chunked_nll` is opt-in for non-assistant-only long-context SFT. |
| DPO | `dpo.yaml` | DPO | Preference-pair training. |
| KTO | `kto.yaml` | KTO | Unlikelihood-style preference training. |
| GKD | `gkd.yaml` | GKD | Text-only distillation with config-driven on-policy ratio, divergence, sequence KD, and dropout policy. |
| GOLD | `gold.yaml` | GKD-style teacher | Uses the upstream TRL experimental trainer. |
| Distillation | `distillation.yaml` | Prompt-only | Stable on-policy distillation with a local teacher and student vLLM generation. |
| GRPO | `grpo.yaml` | GRPO | vLLM importance-sampling correction is enabled by default. |
| async GRPO | `async_grpo.yaml` | GRPO | Requires the trainer/vLLM split runtime. |
| SDPO | `sdpo.yaml` | GRPO-style reward | Uses the upstream TRL experimental trainer. |
| A2PO | `a2po.yaml` | GRPO-style reward | Uses the upstream TRL experimental trainer with ZeRO-2 because its reference-model construction does not support ZeRO-3. |

Distillation resolves its configured version/model-specific vLLM parameter-name profile for both full and PEFT training before the first student weight sync.

Liger is enabled by default for SFT through `training_arguments.use_liger_kernel=True`. Set it to `False` when selecting `chunked_nll` or when native model execution is required.

* Use preprocessed tokenizer option

```shell
is_preprocessed={True or False}
```

* Dataset file path override

```shell
data_path=${connected_dir}/data
dataset_subdir={null or relative directory under data_path}
dataset_file_path={null or full dataset file path}
dataset_file_paths={null or list of dataset file paths}
dataset_files={null or list of {path, format, weight?} source specs}
val_dataset_file_path={null or full validation dataset file path}
val_dataset_file_paths={null or list of validation dataset file paths}
val_dataset_files={null or list of {path, format} validation source specs}
dataset_namespace={null or explicit dataset artifact namespace}
allow_dataset_file_name_mismatch={False or True}
allow_val_dataset_file_name_mismatch={False or True}
test_dataset_file_path={null or full test dataset file path}
test_dataset_file_paths={null or list of test dataset file paths}
test_dataset_files={null or list of {path, format} test source specs}
dataset_resampling.enabled={False or True}
dataset_resampling.strategy={weighted_offline}
dataset_resampling.replacement={False or True}
dataset_resampling.target_size={null or positive integer}
```

`dataset_name` remains the logical dataset family. `dataset_namespace` is the optional, human-readable artifact and logging namespace for a configured dataset composition; when omitted, `effective_dataset_name` falls back to `dataset_name`. By default the train dataset resolves to `${data_path}/${dataset_name}.${dataset_format}`. `dataset_subdir` changes only the directory, `dataset_file_path` is a single-file escape hatch, and `dataset_file_paths` merges same-format files in order. `dataset_files` supports multi-format train sources and still merges by default; set `dataset_resampling.enabled=true` only when weighted offline resampling is required. Multi-file sources require `dataset_namespace`, while their complete source details remain in `resolved_config.yaml` and resolved input metadata instead of being encoded into artifact paths. Explicit `val_dataset_file_path(s)` or `val_dataset_files` disables train-internal validation sampling and uses the provided validation source(s); validation and test sources support multi-format merge only, not weights or resampling. Test data uses `test_dataset_subdir`, `test_dataset_file_path`, `test_dataset_file_paths`, or `test_dataset_files` with the same basename mismatch policy. Default scripts do not need dataset override changes; scripted experiments should override primitive dataset keys and let composed names resolve from config.

* Left padding option

```shell
left_padding={True or False}
```

* Pure decoder based LLM QLoRA 4-bit quantization option

```shell
is_quantized={True or False}
```

* Pure decoder based LLM LoRA or QLoRA PEFT option

```shell
is_peft={True or False}
```

PEFT training uses fresh LoRA initialization by default:

```shell
peft_initialization.mode=fresh
```

To continue training an existing adapter without merging it into the base model, use:

```shell
is_peft=True
peft_initialization.mode=continue_from_adapter
peft_initialization.adapter_path=/path/to/adapter/checkpoint
peft_initialization.adapter_name=default
```

If the adapter path contains `=`, escape each `=` inside the value as `\=` and pass it as a Hydra-quoted value, such as `peft_initialization.adapter_path="/path/with\=a/checkpoint"`.

Adapter continuation is distinct from `resume_from_checkpoint`; it starts a new training run from an existing PEFT adapter, while `resume_from_checkpoint` resumes the same interrupted run. Continuation requires the adapter base model to match the current `pretrained_model_name`, disables merged-model auto-resolution, and incompatible adapter config or unsupported router-LoRA combinations fail fast. `async_grpo` adapter continuation is intentionally unsupported in this release.

* For LLM full fine-tuning(Continued Pretraining) in multi-GPU, recommended

```shell
strategy=deepspeed
```

`async_grpo` uses `strategy=none` for the trainer/vLLM split path.

* GRPO vLLM weight sync strategy

```shell
vllm_sync_strategy={default or lora_streaming}
```

GRPO, SDPO, and Distillation vLLM weight sync resolve parameter-name remapping from `vllm_lora_name_remap`. The default `auto` selection uses model identifiers, modality, and installed package versions, then falls back to the `passthrough` profile. Set `vllm_lora_name_remap.selection` to a configured profile name to force that profile. Each profile may contain multiple non-overlapping prefix rules; unmatched parameter names pass through unchanged.

```yaml
selection: auto
default_profile: passthrough
version_packages: [peft, transformers, trl, vllm]
profiles:
  passthrough:
    prefix_rules: []
  legacy_namespace:
    prefix_rules:
      - source_prefix: model.
        target_prefix: language_model.model.
selectors:
  - name: legacy_model_stack
    model_patterns: [ModelFamily-*]
    modalities: [text]
    package_versions:
      trl: ">=1.0,<2.0"
      vllm: ">=0.10,<0.20"
    profile: legacy_namespace
```

`auto` requires at most one matching selector. Multiple matches fail fast; a concrete `selection` bypasses selector matching. With the pinned dependency stack, Qwen3.5/3.6 text models remap the causal-LM `model.` namespace to vLLM's conditional-generation `language_model.model.` namespace. Qwen3.5/3.6 VLM models use the `passthrough` profile because vLLM's multimodal mapper handles their `model.language_model.` namespace. The resolved profile, selector, package versions, and prefix rules are written to the run manifest.

* GRPO vLLM importance sampling

```shell
vllm_importance_sampling_correction={True or False}
vllm_importance_sampling_mode={sequence_mask}
vllm_importance_sampling_clip_min={null or float}
vllm_importance_sampling_clip_max={null or float}
vllm_importance_sampling_cap={null or float}
```

GRPO vLLM importance-sampling correction is enabled by default with `sequence_mask` mode, `clip_max=3.0`, and no min/cap.

GRPO also enables KL bias correction when `beta` is nonzero. Completion logging is disabled by default. Set both `log_completions=true` and `log_multimodal=true` only when image artifact upload is intended. W&B displays the upstream completion table and images, while MLflow stores each logging event under `completions/tables/` and stores referenced images under `completions/images/`.

* GRPO streaming and agentic training

GRPO keeps map-style local dataset loading by default. Local JSONL, JSON, Parquet, CSV, and TSV files can be streamed without materializing the full dataset by enabling streaming and setting a positive step limit:

```shell
dataset_streaming.enabled=True
max_steps=1000
use_validation=False
```

Streaming is GRPO-only, does not support weighted offline resampling or memory preflight, and requires an explicit validation dataset when `use_validation=True`. TRL preserves each `num_generations` prompt group and uses zero dataset workers for the iterable train stream.

Agentic training is disabled by default. Define task-specific tools and environment factories outside this repository and connect them through Hydra targets:

```yaml
agentic:
  enabled: true
  data_source: environment
  tools: [external_package.tools.lookup]
  environment_factory:
    _target_: external_package.environments.TaskEnvironment
  rollout_worker: null
dataset_namespace: task-environment
max_steps: 1000
use_validation: false
```

`data_source=dataset` keeps the configured GRPO dataset and permits one factory or a name-to-factory mapping. `data_source=environment` skips dataset loading, accepts one factory whose `reset()` owns prompt generation, and requires `dataset_namespace` for artifact isolation, a positive `max_steps`, and disabled validation.

Async GRPO can inject TRL's OpenEnv harness worker through the same config without importing OpenEnv on normal runs:

```yaml
agentic:
  enabled: true
  data_source: dataset
  tools: []
  environment_factory: null
  rollout_worker:
    _target_: trl.experimental.async_grpo.openenv_harness.HarnessRolloutWorker
    harness_session_factory:
      _target_: external_package.openenv.build_session_factory
```

Install the task's OpenEnv client or `openenv[core]>=0.3.1` only for this optional path. The external package owns session construction, task rewards, turn selection, and agent-trace filtering; the repository supplies the configured dataset, tokenizer, reward functions, generation settings, and worker lifecycle inputs.

Async GRPO takes `temperature`, `top_p`, `top_k`, `min_p`, and `repetition_penalty` from `generation_config`. GRPO and Async GRPO use `agentic.max_tool_calling_iterations` for the agentic turn limit, while Async GRPO alone uses `async_agentic.fork_threshold_tokens` for prefix reconciliation. TRL reports reward, completion, throughput, step-time, and queue-time scalar metrics to the selected tracking backend. Raw interactive traces remain task-owned because TRL 1.10 exposes its built-in trace logger only through Trackio.

* SDPO teacher server mode

```shell
teacher_model_kind={ema or live}
use_teacher_server={False or True}
vllm_mode={colocate or server}
vllm_server_base_url={null or http://host:port}
```

SDPO keeps `teacher_model_kind=ema`, `use_teacher_server=False`, and colocate vLLM by default. Teacher-server mode is opt-in and should be used with the upstream live-teacher server requirements: `teacher_model_kind=live`, `use_teacher_server=True`, and `vllm_mode=server`.

For dense-model external vLLM server mode, keep `data_parallel_size=1`.
Current vLLM offline server initialization rejects dense-model `data_parallel_size>1`,
so multi-GPU server utilization should be expressed with tensor parallelism.
This does not change colocate GRPO/SDPO defaults; colocate scripts keep
`vllm_tensor_parallel_size=1` unless the model itself requires tensor-parallel
loading inside the trainer-side vLLM instance.

`QLoRA + GRPO vLLM colocate` is not a supported release target. In colocate mode, TRL initializes vLLM from the quantized trainer model and vLLM stores bitsandbytes-packed weights, while the current LoRA streaming path sends dense LoRA-merged weights. Use non-quantized LoRA for colocate GRPO, or validate a separate external-server path before using QLoRA with vLLM.

* GRPO completion termination override

```shell
completion_termination.enabled={True or False}
completion_termination.terminal_token_texts=[</answer>]
completion_termination.terminal_token_ids=[]
completion_termination.infer_finished_from_short_completion={True or False}
completion_termination.include_model_generation_eos={True or False}
```

`completion_termination` is GRPO-only and disabled by default. It extends TRL's truncated-completion masking beyond tokenizer EOS/PAD by treating configured terminal token ids or token texts as valid completion terminators. `infer_finished_from_short_completion=True` also treats generations shorter than `max_completion_length` as terminated. SDPO and async GRPO do not use this option.

* VLM patch embedding compatibility

| Mode | Behavior |
| --- | --- |
| `native` | Default. Keep the model's original Conv2d or Conv3d patch embedding implementation without probing or mutation. |
| `linear` | Use the equivalent linear projection for structurally compatible full-patch Conv2d and Conv3d modules. Fail when no compatible candidate exists. |
| `auto` | Probe each structural signature in an isolated subprocess and use linear only when correctness passes and native convolution is materially slower or unstable. |

`vision_patch_embedding` is configured in `configs/vision_patch_embedding/base.yaml`. Candidate selection is based on the convolution structure rather than model names or package-version allowlists: groups and padding must preserve independent full patches, dilation must be one, and stride must equal kernel size. Calls that do not contain an exact full patch continue through the original convolution. Auto mode verifies outputs and gradients in float32 and the runtime dtype, benchmarks configured patch counts, treats probe crashes or timeouts as a linear signal only after linear correctness succeeds, and fails on probe infrastructure errors. In distributed runs, one rank selecting linear makes every rank use linear for that signature.

The repository applies the policy to owned Hugging Face primary models and exposed trainer reference or teacher models. GKD/GOLD training, async GRPO, and vLLM-owned test models support `native` only because their relevant model objects are not safely owned by this integration. HF `test` and `test_large` use the same compatibility path as training. `resolved_config.yaml` stores the complete policy, and run or inference manifests store the observed runtime fingerprint, candidate structures, per-rank decisions, applied modules, warnings, and related model roles.

* VLM training image augmentation

```shell
image_augmentation.enabled={True or False}
image_augmentation.backend={pil or albumentations}
image_augmentation.probability={0.0 to 1.0}
image_augmentation.rotation_degrees={degrees}
image_augmentation.jpeg_quality_min={1 to 100}
image_augmentation.jpeg_quality_max={1 to 100}
```

`image_augmentation` is disabled by default and applies only to training images. Validation, evaluation, and test datasets are not augmented. The default backend is `pil`; `backend=albumentations` enables the extended VLM degradation stack and requires `albumentations` plus `opencv-python-headless`. See `configs/image_augmentation/base.yaml` for all controls; `erase_area_min` and `erase_area_max` are area ratios. For bbox/grounding tasks, keep geometry-changing or evidence-removing options such as `rotation_degrees`, `erase_probability`, `albumentations.resize.probability`, and `albumentations.coarse_dropout.probability` disabled unless labels are transformed consistently.

Example Albumentations smoke override:

```shell
image_augmentation.enabled=True
image_augmentation.backend=albumentations
image_augmentation.probability=1.0
image_augmentation.albumentations.seasoning.probability=1.0
```

For Qwen3-VL GRPO with colocated vLLM, small smoke runs may need an explicit `vllm_max_model_length` because the model advertises a very long context window and vLLM sizes KV cache from that value.

GRPO and SDPO expose `steps_per_generation` as an optional generation cadence control. The default `null` preserves TRL behavior, which resolves it to `gradient_accumulation_steps`. Lower values reduce the generation batch size and can reduce peak generation/buffer VRAM, but they usually increase generation frequency and may reduce throughput. Async GRPO does not expose this option in the current TRL API, and GOLD keeps its separate `generation_batch_size` contract because TRL's GOLDConfig does not expose `steps_per_generation`.

VLM image paths are resolved through `dataset_image.image_root_dir` before they reach the processor. Relative paths are interpreted under that root, no-decode paths are normalized to absolute paths, base64 images are decoded to PIL images when they would otherwise be misread as paths, and unsupported direct-path extensions such as `tif`/`tiff` are converted through PIL when `dataset_image.convert_unsupported_extensions=True`.

The table below records VLM training support for the pinned dependency stack. Config overrides remain available, but methods marked text-only are not release-validated for VLM training. The default modality remains `text`.

| Training method | VLM training | Image controls or limitation |
| --- | --- | --- |
| SFT | Yes | `modality`, `max_pixels`, `do_resize`, `image_augmentation`, `dataset_image` |
| DPO | Yes | `modality`, `max_pixels`, `do_resize`, `image_augmentation`, `decode_image_paths`, `dataset_image` |
| KTO | Yes | `modality`, `max_pixels`, `do_resize`, `image_augmentation`, `decode_image_paths`, `dataset_image` |
| GKD / GOLD | No | Text-only repository dataset contract |
| Distillation | Yes | Prompt-only LLM/VLM data with `image` or `images`; student and teacher must share a vocabulary and compatible multimodal inputs. |
| GRPO | Yes | `modality`, `max_pixels`, `do_resize`, `image_augmentation`, `decode_image_paths`, `dataset_image` |
| async GRPO / SDPO / A2PO | No | Text-only with the pinned TRL trainer paths |

* Reward embedding vLLM environment isolation

```shell
reward_embedding.preserved_env_keys=[RANK,WORLD_SIZE,LOCAL_RANK,CUDA_VISIBLE_DEVICES,MASTER_ADDR,MASTER_PORT,NCCL_SOCKET_IFNAME,NCCL_IB_DISABLE,VLLM_WORKER_MULTIPROC_METHOD]
reward_embedding.isolated_env_keys=[RANK,WORLD_SIZE,LOCAL_RANK]
```

* SFT response end template

```shell
response_end_template={null or template such as <|im_end|>}
```

`null` uses the tokenizer EOS token when masking SFT assistant labels.

SFT keeps the repo-owned torch Dataset masking path. `sft_label_mask.validation_mode=strict` fails fast when the response start template is missing, assistant labels are all `-100`, or prompt/padding tokens leak into labels; `report` keeps training unblocked for data inspection.

* Reward extraction profile

```shell
reward.extraction_profile={default or gemma4}
```

`default` keeps existing extraction behavior. `gemma4` strips Gemma channel/turn/tool stop markers before answer extraction. This affects rewards that call `extract_answer_from_generation()`; raw format rewards still check the original completion format.

Reward default hyperparameters and weights are centralized in `configs/reward/base.yaml`; reward class wiring stays in `configs/reward/manager.yaml`.

* KV reward

```shell
reward.weight.kv={0.0 or positive float}
```

`kv` is the only KV reward class for samples whose reward category contains the `kv` token, such as `single_kv` and `multi_kv`. It scores each sample once, so use `reward.weight.kv`. Supported solution roots are `kv`, `tables`, and `results`; predictions may include only the expected top-level root plus keys listed in `reward.kv.allowed_sibling_keys` (`grounding` by default). `results` is matched by `target_id` and scores `results[].text` while ignoring `results[].selected_ids`.

* Grounding bbox reward

```shell
reward.weight.grounding_bbox={0.0 or positive float}
reward.weight.grounding_selection={0.0 or positive float}
reward.grounding_bbox.category_token=bbox
reward.grounding_selection.category_token=evidence
```

`grounding_bbox` is disabled by default. It is evaluated only when the sample reward category contains `reward.grounding_bbox.category_token`.

The label in `solution` should be a JSON object with `grounding_status`, optional `coord_system`, `positive_occurrences` for found targets, and optional `hard_negative_evidence`. Model answers should return JSON with `field_path`, `grounding_status`, and `evidence_occurrences`. Bounding boxes use `[x1, y1, x2, y2]` with page numbers in each occurrence or fragment. Schema aliases are controlled by `reward.grounding_bbox.schema_keys`.

For labels whose `grounding_status` is not `found`, the reward treats an answer with non-`found` status and empty `evidence_occurrences` as the correct negative grounding result.

`grounding_selection` is a separate candidate-selection reward for labels that expect top-level `grounding` lists or compact `target_id -> selected_ids` mappings rather than generated boxes. Each solution and prediction item is matched by `target_id`; item `selected_ids` are compared against the gold item, with the `selected_candidate_ids` alias supported for predictions. It does not use `grounding_status`; value targets are expected to provide evidence candidate ids through `selected_ids`. Missing gold targets, extra predicted targets, duplicate selected ids, wrong candidates, and over-selection are penalized without crashing malformed generations.

For OCR-dependent precise grounding, `selected_ids` follows the inference hook contract: numeric ids and numeric strings are normalized to OCR item ids, boolean ids are invalid, and empty gold selections are skipped as invalid labels. The default cap values preserve the legacy reward shape. Hard grounding runs can lower `partial_match_cap`, `single_id_partial_cap`, `short_multi_id_partial_cap`, `long_multi_id_partial_cap`, `very_long_multi_id_partial_cap`, `over_selection_cap`, `wrong_occurrence_cap`, `schema_only_reward_cap`, `empty_selection_reward_cap`, `invalid_schema_reward_cap`, and `extra_target_reward_cap` so malformed, empty, extra-target, over-selected, or partial outputs cannot keep high reward.

For grounded Auto-KV rows whose solution uses `{"results":[{"target_id":"...","text":"...","selected_ids":[...]}]}<stop>`, use `reward_categories=kv_evidence`, set `reward.weight.kv` and `reward.weight.grounding_selection`, and override `reward.grounding_selection.schema_keys.items=[results]`. This keeps the global grounding default on top-level `grounding` while allowing each results-schema experiment to choose its own reward weights and schema aliases.

* Postprocessing artifact paths

```shell
bash scripts/postprocessing/merge_lora.sh
bash scripts/postprocessing/upload_to_hf_hub.sh
bash scripts/postprocessing/upload_all_to_hf_hub.sh
```

Postprocessing scripts keep `run_id` as a script-local variable. The Python entrypoint resolves `output_dir` from config-composed `output_base_dir` and `run_id`; scripts do not reconstruct checkpoint paths from batch size, devices, gradient accumulation, or timestamps.

* LoRA merge shard size

```shell
merge_max_shard_size={null or shard size such as 6GB}
```

* Pack unpacked Qwen MoE expert tensors after LoRA merge

```shell
merge_pack_qwen_moe_experts={True or False}
```

```shell
python -m src.postprocessing.merge_lora merge_max_shard_size=6GB merge_pack_qwen_moe_experts=True
```

Use `merge_pack_qwen_moe_experts=True` only for Qwen MoE checkpoints saved with unpacked per-expert tensors. Non-Qwen or already-packed checkpoints should keep the default `false`.

Packing rewrites safetensors shards through a temporary directory, so reserve enough disk space for another copy of the merged checkpoint.

* Upload user name and model name at HuggingFace Model card

```shell
upload_user={upload_user} 
model_type={model_type}
```

* Set data and target max length for model training and generation

```shell
max_length={max_length} 
```

__If you want to change main config, use --config-name={config_name}.__

__Also, you can use --multirun option.__

__You can set additional arguments through the command line.__
