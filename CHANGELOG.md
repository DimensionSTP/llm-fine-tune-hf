# Changelog

All notable changes to this repository are documented in this file.

## [v1.23.0] - 2026-06-12

- Add SFT dynamic padding support through `sft_padding_strategy={max_length,dynamic}` while preserving `max_length` as the default behavior.
- Add SFT dataset config wiring for `sft_padding_strategy` and `truncation_mode` across structural and conversational SFT datasets.
- Add `SFTDynamicPaddingCollator` for batch-level right padding of SFT LLM and VLM tensor batches, including sequence tensors and VLM pixel tensors.
- Wire the training pipeline to pass an SFT-specific data collator into trainer construction when dynamic padding is enabled.
- Validate dynamic padding constraints, including unsupported left padding, invalid padding strategy values, invalid truncation modes, empty batches, inconsistent batch keys, and non-tensor batch values.
- Record the active SFT padding strategy in run metadata for reproducibility and auditability.
- Decode base64 VLM image payloads before path handling so encoded images are not misread as filesystem paths.
- Export the SFT dynamic padding collator API for downstream reuse.
- Document SFT padding strategy options, `pad_to_multiple_of` behavior, right-padding limits, and base64 VLM image handling in README and usage guide updates.

## [v1.22.0] - 2026-06-11

- Add PEFT adapter continuation support so a new PEFT training run can start from an existing LoRA adapter without first merging the adapter into the base model.
- Add `configs/peft_initialization/lora.yaml` and compose the PEFT initialization defaults into SFT, DPO, KTO, GKD, GRPO, SDPO, and Async GRPO configs.
- Wire model setup to choose between fresh LoRA initialization and `continue_from_adapter` initialization through the shared `peft_initialization` config.
- Validate adapter continuation inputs, including required adapter path, trainable adapter mode, base-model match enforcement, unsupported router-LoRA combinations, and intentionally unsupported Async GRPO continuation.
- Keep adapter continuation separate from interrupted-run resume behavior by bypassing merged-model auto-resolution when continuing from an adapter.
- Record PEFT initialization metadata in run manifests, including mode, adapter identity, base-model references, config fingerprints, and continuation-related compatibility details.
- Export PEFT initialization helper APIs for reuse by model loading, setup, run metadata, and downstream integrations.
- Document PEFT adapter continuation usage, Hydra escaping for adapter paths, and the difference between adapter continuation and `resume_from_checkpoint`.

## [v1.21.0] - 2026-06-11

- Add dataset path resolver support for `data_path`, `dataset_subdir`, `dataset_file_path`, and filename mismatch guard settings across training and test datasets.
- Route SFT, DPO, KTO, GKD, GRPO, and test dataset loaders through resolved dataset paths so dataset location overrides use one shared contract.
- Add shared dataset image IO helpers and dataset image config defaults for image root resolution, unsupported extension conversion, and image mode normalization.
- Normalize VLM image inputs in dataset loaders, including path, decoded image, and converted image handling for training and test flows.
- Add SFT label-mask validation with strict defaults for assistant label token coverage and truncated-assistant reporting.
- Record resolved dataset paths in runtime metadata for reproducibility and run auditing.
- Centralize reward defaults in `configs/reward/base.yaml` and compose them into GRPO, SDPO, and Async GRPO configs.
- Add configurable grounding bbox schema/status mapping and new `GroundingSelectionReward` support.
- Stabilize KV reward table handling and export the new dataset helper and reward APIs.
- Sync grounding selection reward overrides in GRPO, SDPO, and Async GRPO launcher scripts.
- Update README, reward documentation, and usage guide coverage for dataset path, image loading, SFT mask validation, and reward configuration changes.

## [v1.20.2] - 2026-06-05

- Add `configs/run_metadata/allocation.yaml` with configurable distributed run allocation timeout, poll interval, and freshness grace settings.
- Compose the run metadata allocation config into SFT, DPO, KTO, GKD, GRPO, SDPO, and Async GRPO training configs.
- Validate distributed run allocation payload freshness so non-rank0 processes ignore stale allocation files from previous runs.
- Include allocation key and output base directory metadata in shared run allocation payloads to prevent cross-run allocation reuse.
- Write JSON metadata through a temporary file and atomic replace so readers do not consume partially written allocation or run metadata files.
- Document run metadata allocation settings for distributed and multi-node training.

## [v1.20.1] - 2026-06-04

- Add Async GRPO vLLM tensor-parallel-size resolution so `async_runtime.vllm_server.tensor_parallel_size=auto` maps to the GPU count assigned to the vLLM server side.
- Change the Async GRPO vLLM server tensor-parallel default from `1` to `auto` in the main config and Async GRPO launcher scripts.
- Validate explicit Async GRPO vLLM tensor-parallel sizes so they cannot exceed the number of assigned vLLM GPUs.
- Derive external GRPO vLLM server tensor-parallel size from `gpu_ids` while keeping dense-model `data_parallel_size=1`.
- Document Async GRPO and external GRPO vLLM tensor-parallel policy, including dense-model external-server constraints and colocate-mode expectations.

## [v1.20.0] - 2026-06-03

- Add distributed runtime planning, validation, and manifest metadata for single-node and multi-node training, including planned/observed world size, local process count, device selection, and effective batch-size reporting.
- Add distributed config defaults across SFT, DPO, KTO, GKD, GRPO, SDPO, and Async GRPO with `distributed.enabled`, `num_machines`, `num_processes_per_machine`, `machine_rank`, `main_process_ip`, `main_process_port`, and `validation_mode`.
- Add a shared model-loading policy planner and `configs/model_loading/train_runtime.yaml` to centralize DeepSpeed ZeRO-3 initialization, QLoRA device-map handling, inference device maps, and FSDP validation rules.
- Add managed Async GRPO split-runtime support for trainer/vLLM separation, including rank-specific runtime handling, stop-signal coordination, isolated vLLM server environment setup, readiness checks, and configurable vLLM server flags.
- Add multi-node launch scripts for SFT, DPO, KTO, GKD, GRPO colocate, GRPO external-server, Async GRPO, and SDPO training.
- Add a local SDPO trainer wrapper that supports PEFT EMA teacher updates under ZeRO-3 while preserving TRL SDPO trainer behavior for non-PEFT paths.
- Add conditional chat-template thinking kwargs so `enable_thinking` is passed only when the active tokenizer or processor chat template supports it.
- Add DPO preference dataset image-path decoding support and extend existing GRPO image payload decoding to handle dictionary payloads with image bytes or paths.
- Fix LoRA streaming vLLM weight synchronization for Qwen3.5/Qwen3.6-style language-model prefixes and quantized base weights.
- Expose vector reward weights in GRPO and SDPO launcher scripts and update default training/script values for Qwen3.5/Qwen3.6 model families.
- Document multi-node launchers, distributed runtime manifest metadata, Async GRPO split-runtime expectations, and GRPO vLLM/QLoRA compatibility limits.

## [v1.19.0] - 2026-06-01

- Add SFT `sft_loss_type` configuration with `nll` and `chunked_nll` options, wire it into SFT training arguments, and validate that `chunked_nll` is not combined with Liger kernel execution.
- Add automatic train `run_id` allocation so training outputs are written under `${output_base_dir}/${run_id}` and resume flows recover `run_id`, `output_base_dir`, and `output_dir` from the resumed checkpoint path.
- Add training run metadata artifacts, including `run_manifest.json`, `resolved_config.yaml`, and `training_args.json`, before model construction for reproducibility and downstream auditability.
- Move runtime batch-size fields out of checkpoint path names and into run metadata while preserving method, model, dataset, strategy, PEFT, length, and active reward information in `save_detail`.
- Add `output_base_dir`, runtime-populated `output_dir`, and nullable `run_id` defaults across SFT, DPO, KTO, GKD, GRPO, SDPO, and Async GRPO configs.
- Update postprocessing merge and Hugging Face Hub upload entrypoints to resolve existing artifacts from `output_base_dir` plus `run_id`, and update scripts to call module entrypoints instead of reconstructing checkpoint paths from batch fields.
- Update Hugging Face Hub token access to the current `get_token()` API across upload and scaling utilities.
- Simplify reward save suffix generation to derive path suffixes from active reward keys.
- Declare Qwen3.5 runtime dependencies `causal-conv1d==1.6.2.post1` and `flash-linear-attention==0.5.0` in both direct dependency files.
- Document `--no-build-isolation` install commands, SFT loss-type selection, run artifact workflow, and the updated training/postprocessing artifact contract.

## [v1.18.0] - 2026-05-22

- Add disabled-by-default VLM training image augmentation configuration under `configs/image_augmentation/base.yaml`.
- Support train-only image augmentation for SFT, DPO, GRPO, Async GRPO, and SDPO dataset pipelines while leaving validation, evaluation, and test image processing unaugmented.
- Add configurable image transforms for rotation, JPEG compression, Gaussian blur, contrast, brightness, sharpness, grayscale conversion, noise, erasure, and ink-bleed simulation.
- Preserve existing image resize behavior when augmentation is disabled and combine augmentation with resize processing when augmentation is enabled.
- Document VLM image augmentation options, safety guidance for bbox/grounding labels, and common CLI override keys in README and the usage guide.
- Declare `pillow==12.1.1` as a direct dependency in `requirements.txt` and `pyproject.toml`.

## [v1.17.0] - 2026-05-21

- Add a GRPO completion termination patch that treats configured terminal token ids, terminal token texts, tokenizer EOS/PAD, and optionally model generation EOS as valid completion terminators.
- Support `completion_termination.infer_finished_from_short_completion` so completions shorter than `max_completion_length` can be treated as finished instead of truncated.
- Wire the completion termination patch into the GRPO training pipeline while leaving SDPO and Async GRPO unaffected.
- Add disabled-by-default `completion_termination` settings to `configs/grpo.yaml`.
- Export `patch_grpo_completion_termination` from `src.utils` and document GRPO completion termination options in README.

## [v1.16.0] - 2026-05-21

- Add `GroundingBBoxReward` for page-level bounding-box grounding rewards with schema, page-match, IoU, center-in-box, large-box, hard-negative, and duplicate-positive shaping.
- Export `GroundingBBoxReward` from `src.utils` for reward manager construction and downstream imports.
- Wire grounding bbox reward configuration through GRPO, Async GRPO, SDPO, and the reward manager, keeping `reward.weight.grounding_bbox` disabled by default.
- Document grounding bbox reward usage, label/prediction schema, activation by reward category token, and non-found grounding behavior in README and `REWARDS.md`.
- Preserve the intended SFT dataset config EOF discipline.

## [v1.15.0] - 2026-05-20

- Add configurable SFT response-end masking through `response_end_template`, allowing assistant-label masking to use either tokenizer EOS or an explicit response terminator.
- Wire `response_end_template` through SFT dataset configs and the main SFT config defaults.
- Add reward extraction profiles with `default` and `gemma4` handling so answer extraction can normalize Gemma channel, turn, and tool stop markers before reward scoring.
- Wire `reward.extraction_profile` through GRPO, Async GRPO, SDPO, and reward manager configs for all reward classes that use answer extraction.
- Document SFT response-end masking and reward extraction profile options in README and the usage guide.
- Restore trailing newlines in updated config files to keep YAML files editor- and tooling-friendly.

## [v1.14.0] - 2026-05-18

- Add LoRA merge output sharding support so merged checkpoints can be saved with configurable `merge_max_shard_size` limits.
- Add optional Qwen MoE expert tensor packing for LoRA-merged checkpoints saved with unpacked per-expert tensors.
- Add `merge_max_shard_size` and `merge_pack_qwen_moe_experts` defaults across SFT, DPO, KTO, GKD, SDPO, GRPO, and Async GRPO configs.
- Document LoRA merge sharding and Qwen MoE expert packing options in README and the usage guide.

## [v1.13.1] - 2026-05-14

- Use reentrant gradient checkpointing by default across SFT, DPO, KTO, GKD, SDPO, GRPO, and Async GRPO training configs.
- Define explicit `decode_image_paths: false` defaults for Async GRPO and SDPO dataset configs so image path decoding behavior is stable unless enabled intentionally.
- Disable DeepSpeed memory breakdown logging in the default DeepSpeed config to avoid unnecessary memory-breakdown collection during standard runs.
- Sync `packages.txt` with the current validated runtime freeze, including `mpi4py==4.1.1` and `mpich==5.0.1` in the full environment snapshot.
- Keep direct install dependency pins unchanged because the newly recorded MPI packages are not imported by the project code or required as direct package dependencies.

## [v1.13.0] - 2026-05-14

- Add configurable GRPO vLLM sync strategy support, including the `lora_streaming` strategy for streaming LoRA-merged weights into vLLM without full adapter merge/unmerge cycles.
- Add reward embedding vLLM environment isolation controls so retrieval rewards can initialize vLLM without inheriting trainer rank state, then restore the preserved distributed environment.
- Construct training arguments with the resolved DeepSpeed config at instantiation time so trainer setup receives the expected distributed strategy configuration.
- Guard colocated vLLM graph recapture by training method and clean up colocated runtime helper internals.
- Disable GRPO `bf16_full_eval` by default to avoid full-evaluation dtype issues in the current GRPO runtime path.
- Compact reward name formatting and refresh README, usage guide, and reward documentation for the new vLLM runtime and reward embedding options.

## [v1.12.0] - 2026-05-12

- Add colocated vLLM runtime utilities for graph recapture handling before colocated trainer execution.
- Export colocated vLLM runtime helper APIs from `src.utils` for reuse by GRPO training paths.
- Prepare reward vLLM models before colocated GRPO training so colocated runtime state is ready before trainer execution.
- Sync `packages.txt` with the current validated runtime freeze after the training-stack refresh.
- Update install-time and runtime pins for `huggingface-hub==1.14.0`, `numpy==2.4.4`, `transformers==5.8.0`, `trl==1.4.0`, and `vllm==0.18.0` across `requirements.txt` and `pyproject.toml`.
- Preserve the existing `flash-attn` direct Git install path while aligning the core LLM/VLM training dependency manifests with the validated environment.

## [v1.11.0] - 2026-05-11

- Add configurable GRPO image source and image path decoding for multimodal dataset loading.
- Expose image path decoding settings through `configs/grpo.yaml` and `configs/dataset/grpo.yaml`.
- Decode configured GRPO image sources during dataset construction so image paths can be resolved before training.

## [v1.10.0] - 2026-04-07

- Add full Async GRPO training support with new config set, training-arguments profile, trainer profile, runtime lifecycle utility, and `scripts/train/async_grpo_train.sh`.
- Add SDPO training support with new method config, training-arguments profile, trainer profile, and `scripts/train/sdpo_train.sh`.
- Wire async GRPO server lifecycle handling into the main training pipeline and refresh related documentation for runtime behavior.
- Align existing method/trainer/training-argument configs with TRL 1.0 schema expectations across GRPO, GKD, DPO, and SFT paths.
- Improve multimodal training compatibility by supporting `pixel_position_ids` in VLM collate functions.
- Add reward adapter logging outputs and refine reward/hydra resolver wiring used by the updated trainer paths.
- Reorder and extend exported utility surfaces to support the new async runtime and method entrypoints.

## [v1.9.2] - 2026-04-06

- Sync `packages.txt`, `requirements.txt`, and `pyproject.toml` with the current validated runtime freeze after compatibility-driven dependency adjustments.
- Update direct packaging dependency pins (including `vllm`-line compatibility levels) so install-time and runtime manifests stay consistent.

## [v1.9.1] - 2026-04-03

- Sync `packages.txt`, `requirements.txt`, and `pyproject.toml` with the current validated runtime environment.
- Update packaging metadata and direct dependency pins so install-time and runtime dependency manifests stay aligned.

## [v1.9.0] - 2026-03-27

- Add configurable negative reward penalties for GRPO `MatchReward` and `CodeExecutionReward`.
- Wire negative-penalty settings through `configs/grpo.yaml` and `configs/reward/manager.yaml`.
- Include active negative-penalty settings in Hydra artifact suffix generation and reward metric naming for clearer experiment tracking and logging.
- Support the current Qwen3 packed-expert MoE layout in dense-to-MoE preprocessing and verification.
- Extend dense-LoRA-to-MoE merge and verification logic to support packed-expert MoE layouts.
- Add sparse-decoder dense-to-MoE preprocessing and verification entrypoints, and export stable public aliases from `src.preprocessing`.
- Add vLLM sync helpers for router-with-lora sparse MoE trainers and apply them in GRPO pipeline wiring when compatible checkpoints are used.
- Harden model setup loading for Transformers 5.3 compatibility.
- Ignore `*.stdout` and `*.stderr` nohup-style outputs in Git.

## [v1.8.0] - 2026-03-25

- Add `reward.retrieval.ndcg.weighting_mode` to support configurable nDCG cutoff emphasis for retrieval rewards.
- Support both `small_k` (`k^-alpha`) and `large_k` (`k^alpha`) weighting modes in `RetrievalnDCGReward`.
- Update the default GRPO retrieval nDCG configuration to use `weighting_mode: large_k`.
- Include `weighting_mode` in reward naming and Hydra reward-save suffix generation for clearer experiment identification.
- Update reward documentation to explain the new weighting behavior and current default.

## [v1.7.3] - 2026-03-24

- Remove stale references to deleted Korean documentation files from the training and evaluation contract.
- Align release/documentation sync guidance with the current English documentation set.
- Prepare a patch release for documentation-contract consistency after KO document removal.

## [v1.7.2] - 2026-03-23

- Remove outdated Korean docs (`*_ko.md`) to prevent EN/KO content drift; keep EN docs as canonical source for now.
- Revert Hydra entry-point defaults to prior execution behavior for W&B local directory compatibility.
- Keep `flash-attn` as optional GPU dependency path and align docs accordingly.

## [v1.7.1] - 2026-03-19

- Add collaboration metadata: `CONTRIBUTING.md`, `SECURITY.md`, `CODEOWNERS`.
- Add GitHub templates: PR template, issue templates, and docs/link CI workflow.
- Add `.env.example` template for onboarding and local setup.
- Add Python compile smoke workflow: `.github/workflows/python-compile-check.yml`.
- Add execution contract document: `TRAINING_EVAL_CONTRACT.md`.
- Align `pyproject.toml` version with release line (`1.7.0`).
- Patch release for packaging and execution-governance improvements with updated release notes.
- Refer to the GitHub Release note for full details and migration context.
