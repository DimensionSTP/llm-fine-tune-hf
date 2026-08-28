# Changelog

All notable changes to this repository are documented in this file.

## [v2.10.8] - 2026-08-28

- Add a dedicated MLflow operations guide covering local and remote tracking profiles, environment and authentication setup, training and evaluation launches, preflight checks, and run identity.
- Document active-run monitoring, search and comparison workflows, MLflow 3.13 line smoothing, artifact and tag inspection, distributed training, and Slack lifecycle notifications.
- Document safe resume and rerun procedures, Hydra override escaping, terminal run states, troubleshooting, and an end-to-end operational checklist without changing runtime behavior.

## [v2.10.7] - 2026-08-27

- Redact credential-bearing values across the complete inference companion manifest, including resolved configuration, resolved inputs, runtime metadata, and nested extension payloads, before atomic persistence.
- Preserve canonical inference result and companion paths, non-sensitive reproducibility metadata, runtime configuration, and all existing Transformers, large-model, vLLM, multi-turn, PEFT adapter, and vision patch embedding behavior.

## [v2.10.6] - 2026-08-26

- Redact credential-bearing config values, propagated secrets, URI credentials, CLI overrides, failure messages, and persisted memory-preflight commands without changing runtime config or subprocess arguments.
- Validate all available training metadata before MLflow upload, fail closed without partial metadata upload, and preserve the original failure or interrupt while logging only redacted secondary errors.
- Disable duplicate Hydra metadata persistence and document environment-only secret delivery and canonical redacted metadata handling.

## [v2.10.5] - 2026-08-26

- Record package, model, modality, strategy, artifact run, and Git revision metadata as searchable MLflow tags for fresh and resumed runs.
- Upload available training lifecycle metadata before finalizing MLflow runs as `FINISHED`, `FAILED`, or `KILLED`, preserving completed, failed, and interrupted run evidence under the `metadata/` artifact path.

## [v2.10.4] - 2026-08-24

- Extend config-driven vLLM parameter-name remapping to offline PEFT `test_vllm` and `test_vllm_multi_turn` execution while preserving the existing training weight-sync behavior across all supported methods.
- Validate immutable local safetensors adapters before vLLM model loading and create or reuse content-addressed remapped adapters without modifying the source checkpoint.
- Record source adapter hashes and effective vLLM adapter path, hash, tensor counts, and materialization action in inference companion manifests.
- Use a valid positive LoRA request ID for multi-turn vLLM inference and validate Qwen3.5 text PEFT adapter remapping, generation, artifact reuse, source immutability, and process cleanup through focused CPU and GPU gates.

## [v2.10.3] - 2026-08-23

- Propagate the existing `is_enable_thinking` policy through row-level chat-template arguments for DPO, KTO, GKD, GOLD, and A2PO training and validation datasets while preserving trainer-level handling for GRPO, SDPO, Async GRPO, and Distillation.
- Filter row-level chat-template arguments against the active tokenizer or processor template while preserving DPO and KTO VLM image transforms and collator inputs.

## [v2.10.2] - 2026-08-21

- Add the pinned cuRAND runtime and development packages required to build DeepSpeed CPUAdam with the CUDA 12.9 environment.
- Update Flash Linear Attention to 0.5.2 to apply the Blackwell Gated DeltaNet backward autotune restriction and prevent CUDA misaligned-address failures.
- Synchronize the project metadata, requirements, environment freeze, and installation guidance without changing the validated PyTorch, Transformers, TRL, vLLM, or FlashInfer runtime versions.

## [v2.10.1] - 2026-08-21

- Add the CUDA 12.9 compiler installation contract and pinned FlashInfer 0.6.14 CUDA 12.9 JIT cache required for Blackwell sampler initialization when the system CUDA toolkit is older than CUDA 12.9.
- Add `SLACK_WEBHOOK_URL` to `.env.example` so the default Slack notification backend can be configured directly from the onboarding environment template.
- Synchronize the project metadata, requirements, environment freeze, and installation guidance without changing the validated PyTorch, Transformers, TRL, vLLM, or FlashInfer runtime versions.

## [v2.10.0] - 2026-08-21

- Update the validated training stack to PyTorch 2.11.0 with CUDA 12.9, Transformers 5.14.1, TRL 1.10.0, vLLM 0.26.0+cu129, MLflow 3.13.0, W&B 0.25.1, FlashInfer 0.6.14, and the synchronized candidate environment freeze.
- Add a reproducible CUDA 12.9 installation path, preserve pandas 3.0.1 through an explicit requirements override, and keep project, requirements, and package-freeze dependency records synchronized.
- Make MLflow the default tracking backend while preserving config-only switching to W&B, shared run identity, lifecycle finalization, train/eval metrics, completion artifacts, and backend-specific resume behavior.
- Add backend-neutral lifecycle notifications with Slack as the default notification profile and keep W&B alerts available as an explicit notification backend.
- Scope MLflow GPU system metrics to `CUDA_VISIBLE_DEVICES`, attach GRPO completion tables and images to tracking runs, and prevent immutable training parameters from being re-logged when resuming an existing MLflow run.
- Add stable TRL Distillation training with dedicated datasets, configs, trainer wiring, memory preflight, single-node and multi-node launchers, teacher-model controls, and colocated or external vLLM inference support.
- Add config-driven agentic GRPO and Async GRPO inputs with optional tools, environments, turn limits, queue controls, sampling settings, validation, runtime metadata, and disabled-by-default compatibility.
- Add streaming GRPO dataset support that preserves completion groups, enforces explicit validation data, disables incompatible workers and memory preflight, and terminates correctly under `max_steps`.
- Expose TRL 1.10 GRPO controls for bias correction, importance sampling, multimodal completion logging, sampling, and maximum vLLM model length.
- Adapt SFT datasets to the updated TRL dataset contract while preserving nested multimodal fields, image tensors, vision patch embedding gradients, fixed and dynamic padding behavior, and actual visual-parameter updates.
- Add targeted TRL runtime compatibility patches for GKD gradient checkpointing, SDPO ZeRO-3 EMA adapters, A2PO ZeRO-2 reference training, and supported trainer argument surfaces.
- Generalize config-driven vLLM parameter-name remapping across GOLD, GRPO, Async GRPO, SDPO, and Distillation while retaining passthrough behavior for unmatched models and runtimes.
- Remap Qwen3.5 text-model weights to the `language_model.model.*` vLLM namespace for colocated, server, streaming, and packed Async GRPO weight-transfer paths.
- Isolate Async GRPO trainer and managed vLLM GPUs before CUDA initialization, restart with the active Python interpreter when required, preserve non-main training ranks, and cleanly coordinate server readiness and shutdown.
- Replace deprecated warmup-ratio trainer wiring with the unified `warmup_steps` contract while preserving fractional warmup ratios and absolute integer warmup steps.
- Preserve explicitly selected physical GPUs during distributed memory-preflight subprocess launches instead of remapping inherited `CUDA_VISIBLE_DEVICES` values to unrelated local indices.
- Synchronize training configs, launchers, public exports, README guidance, usage documentation, and the training/evaluation contract with the updated tracking, Distillation, GRPO, vLLM, and runtime behavior.
- Validate the release stack through single-GPU coverage plus a final 13-case acceptance matrix covering memory preflight, two-node training, checkpoint resume, multi-GPU VLM GRPO, MLflow, W&B switching, and agentic Async GRPO with no failed cases.

## [v2.9.0] - 2026-08-14

- Replace implicit multi-file dataset naming with an explicit `dataset_namespace` contract shared by every training method, while retaining `dataset_name` as the fallback artifact namespace.
- Narrow and synchronize dataset input metadata so complete source lineage remains in resolved metadata without leaking file composition details into artifact paths.
- Expand training manifests into a complete `prepared`, `running`, `completed`, `failed`, or `interrupted` lifecycle with stage, failure, runtime, PEFT, dataset, and relative artifact evidence.
- Bind memory-preflight evidence to its parent training run, preserve dataset metadata for probe subsets, and prevent preflight execution from allocating an independent run or tracking identity.
- Persist and validate the selected tracking backend alongside the backend run id so resumed runs cannot silently switch tracking systems.
- Preserve canonical `${dataset_name}` inference result names, remove the redundant `test_output_name` setting, and simplify inference and dense-LoRA companion manifests around full resolved config plus observed runtime evidence.
- Add structural Conv2d and Conv3d vision patch embedding compatibility with `native`, `linear`, and isolated `auto` probe modes for supported Hugging Face VLM training and test paths.
- Validate vision patch candidates, output and gradient equivalence, distributed candidate inventories, probe timeouts, and slowdown-ratio decisions before applying compatible linear projections.
- Record vision patch compatibility plans, decisions, model roles, runtime fingerprints, applied modules, warnings, and probe evidence in training and inference manifests.
- Preserve VLM image tensors and auxiliary processor fields through fixed and dynamic SFT collation.
- Enable dynamic padding and the Liger kernel by default for SFT while retaining explicit fixed-padding and native-kernel overrides and rejecting unsupported `chunked_nll` combinations.
- Align DeepSpeed checkpoint behavior with standard save and resume operation by disabling universal checkpoint loading and removing universal checkpoint saving.
- Add `nvidia-ml-py==13.595.45` for NVIDIA runtime inspection and keep project and requirements dependency pins synchronized.
- Update the README, usage guide, and training/evaluation contract for dataset namespaces, artifact lifecycle, canonical inference names, vision patch modes, SFT defaults, DeepSpeed behavior, and the release-validated VLM method matrix.

## [v2.8.1] - 2026-08-12

- Register the repository's custom Hydra resolvers before composing configuration in the LoRA merge and Hugging Face Hub upload entrypoints, restoring postprocessing execution for configs that reference `dataset_effective_name` or `reward_save_suffix`.

## [v2.8.0] - 2026-08-12

- Isolate training checkpoints by `effective_dataset_name` across all supported fine-tuning methods so transformed or mixed dataset variants cannot collide under the same artifact namespace.
- Add a versioned training manifest lifecycle that records `prepared` before execution and transitions atomically to `completed` only after training and model saving succeed.
- Record resolved training artifact paths and existence states in `run_manifest.json`, and finalize that metadata after `trainer.save_model()` completes.
- Add companion inference manifests for Transformers, large-model, vLLM, and multi-turn vLLM test modes with model, adapter, dataset, runtime, generation, and result lineage.
- Compose test artifact directories from `model_detail`, give multi-turn results an explicit `_multi_turn` suffix, and remove launcher-level output-path overrides so configuration remains the single path authority.
- Add a shared tracking lifecycle that maps normal completion, ordinary exceptions, and interrupts to `FINISHED`, `FAILED`, and `KILLED` MLflow terminal states without allowing alert or finalization failures to replace the original pipeline error.
- Apply lifecycle-managed tracking consistently to training and every test pipeline while preserving rank-zero ownership in distributed execution.
- Add a remote MLflow server profile driven by `MLFLOW_TRACKING_URI`, optional basic-auth environment variables, and fail-fast validation when the required server URI is missing.
- Export the new artifact metadata and tracking lifecycle utilities through the public `src.utils` API.
- Document checkpoint namespaces, train and inference manifest contracts, test artifact layouts, and local versus remote MLflow operation in the README, usage guide, and training/evaluation contract.

## [v2.7.1] - 2026-08-10

- Correct the shebangs in the dense-to-MoE preprocessing launchers so direct shell execution resolves `/bin/bash` correctly.
- Remove unsupported `is_sft` overrides from single-node and multinode GKD launchers, keeping their CLI surface aligned with `configs/gkd.yaml`.
- Align single-node and multinode GRPO launcher clipping values with `configs/grpo.yaml` by using `epsilon=0.2` and `epsilon_high=null`.
- Align single-node and multinode asynchronous GRPO launcher clipping values with `configs/async_grpo.yaml` by using `epsilon=0.2` and `epsilon_high=0.2`.
- Align all standard, large, and vLLM test launcher adapter metadata with the configured PEFT defaults `r=128` and `lora_alpha=512` so adapter paths resolve to the trained artifacts.
- Remove an unnecessary blank line from the `BaseReward` class body to keep its layout consistent with repository formatting conventions.
- Replace the quoted local `_DatasetBuilder` annotation with a direct protocol reference, simplifying the type declaration without changing dataset construction behavior.

## [v2.7.0] - 2026-08-10

- Add configurable vLLM LoRA streaming parameter-name remap profiles with automatic or explicit selection based on model identifiers, modality, and runtime package versions.
- Add a pinned-stack profile that remaps Qwen3.5 and Qwen3.6 text-model `model.` parameters to the vLLM conditional-generation `language_model.model.` namespace while preserving passthrough behavior for VLM and unmatched runtimes.
- Add fail-fast validation for remap profiles, prefix rules, selectors, package declarations, version specifiers, profile references, and ambiguous selector matches.
- Resolve the active remap profile before run artifact capture and record the selected profile, selector, package versions, and prefix rules in run metadata.
- Add `packaging==26.2` to project and requirements dependencies for standards-compliant runtime version matching.
- Isolate training-argument validation inputs from memory-preflight mutation, remove an unused MoE verification result key, and simplify LoRA merge status reporting.
- Reorganize and export dataset image, postprocessing, preprocessing, reward, scaling, asynchronous GRPO, Hydra resolver, memory preflight, PEFT initialization, run metadata, setup, and vLLM synchronization APIs according to execution and public-use order.
- Normalize multi-argument call formatting across datasets, pipelines, preprocessing, rewards, scaling, and utilities, and enforce repository YAML file-ending conventions for GitHub metadata and workflows.
- Document vLLM LoRA remap configuration, automatic selection behavior, Qwen text and VLM namespace handling, explicit profile selection, failure conditions, and run-manifest fields.

## [v2.6.0] - 2026-07-15

- Update Hugging Face training dependency pins for the current runtime stack, including `transformers==5.13.1`, `trl==1.8.0`, `vllm==0.19.1`, `safetensors==0.8.0`, and `compressed-tensors==0.15.0.1` across dependency files.
- Fix KTO configuration targets for the current TRL public export layout and expose SFT dataset column names required by the updated TRL trainer path.
- Repair command continuations in the small and large test launcher scripts so chained smoke commands preserve the intended shell structure.
- Introduce the `src.rewards` package with shared base utilities and a package-level reward manager, then split format, text, retrieval, KV, grounding, embedding, vector-store, and vLLM runtime reward logic out of the legacy monolithic rewards module.
- Move reward Hydra targets to the exported `src.rewards.*` API and remove reward exports from `src.utils`, keeping config paths aligned with the new package boundary.
- Add grounded KV reward support for allowed sibling payload fields so `KVReward` can score configured KV fields while tolerating grounding-side evidence fields such as selected ids in the same result objects.
- Wire the allowed-sibling KV controls through `configs/reward/base.yaml` and `configs/reward/manager.yaml`, and document the grounded KV compatibility contract in the README and reward reference.

## [v2.5.1] - 2026-07-03

- Add `KVReward` support for top-level `results` solution roots so grounded Auto-KV rows can score extracted text by `target_id`.
- Allow solution JSON with the configured KV stop token to be parsed before reward scoring, matching grounded Auto-KV labels that serialize as JSON followed by the stop marker.
- Score `results[].text` while ignoring `results[].selected_ids` in the KV reward path so evidence ids can be evaluated separately by the grounding selection reward.
- Preserve existing `kv` and `tables` root scoring behavior while adding results-specific root, length, structure, and content scoring helpers.
- Document the grounded Auto-KV reward contract in the README and reward reference, including the `kv_evidence` category, `reward.weight.kv`, `reward.weight.grounding_selection`, and `reward.grounding_selection.schema_keys.items=[results]` setup.

## [v2.5.0] - 2026-06-25

- Add strict memory preflight support that can run a subprocess probe before training and fail fast on out-of-memory conditions before the main run starts.
- Add `configs/memory_preflight/base.yaml` and compose the disabled-by-default `memory_preflight` policy into SFT, DPO, KTO, GKD, GRPO, Async GRPO, SDPO, A2PO, and GOLD configs.
- Select max-shape probe samples, persist selected probe indices, isolate probe outputs under `.memory_preflight`, and record memory preflight settings in run metadata.
- Support single-node distributed memory preflight coordination for direct launches while rejecting unsupported multi-node, vLLM server, and async topologies.
- Add PEFT `target_parameters` handling, validation, metadata export, and public utility exports for target-parameter-aware adapter initialization.
- Update model loading planning so ZeRO-3 loading remains compatible with PEFT target parameters by requiring `zero3_init_method=auto` or `disabled` when target parameters are configured.
- Document memory preflight usage, failure behavior, supported topology constraints, and the relationship between probe output paths and normal training artifacts.
- Normalize tracked shell scripts to the repository EOF policy without changing their launcher behavior.

## [v2.4.0] - 2026-06-23

- Add centralized `dataloader_runtime` configuration and resolver support so train and test loaders derive worker, prefetch, persistent-worker, and pin-memory settings from shared runtime policy instead of script-level overrides.
- Record resolved dataloader runtime settings in run metadata, remove legacy dataloader method defaults from training arguments, and simplify train/test launcher scripts to rely on config composition.
- Add GRPO and SDPO `steps_per_generation` controls with metadata recording and README guidance for generation cadence, VRAM, and throughput tradeoffs.
- Add multi-dataset path resolution, multi-format dataset loading, optional weighted offline resampling, and dataset identity metadata for train, validation, and test inputs.
- Add shared dataset-loading helpers and wire multi-dataset input support across SFT, DPO, KTO, GKD, GRPO, and test dataset classes with exported helper APIs.
- Strengthen KV reward structure scoring with stricter defaults for malformed JSON, stop-token handling, root-shape mismatches, serialized length, leaf-count imbalance, duplicate values or paths, table structure, and non-exact value matches.
- Harden grounding selection reward scoring with configurable caps for partial matches, over-selection, wrong candidates, duplicate selections, invalid schemas, empty selections, and extra predicted targets.
- Expose grounding selection reward controls through `configs/reward/base.yaml` and `configs/reward/manager.yaml`, and document the selection-reward contract in the README.
- Remove AGENTS-disallowed implementation comments while preserving the updated dataset, dataloader, reward, and metadata behavior.

## [v2.3.1] - 2026-06-19

- Change the default SFT loss type from `chunked_nll` to `nll` so assistant-only SFT (`is_sft=True`) uses the supported non-chunked loss path by default.
- Document `chunked_nll` as an opt-in long-context SFT loss path for non-assistant-only SFT, and note that `chunked_nll` should not be combined with `is_sft=True` or Liger kernel execution.
- Change default PEFT adapter names from `adapter` to `default` across PEFT initialization and PEFT test configs for SFT, DPO, KTO, GKD, GRPO, Async GRPO, SDPO, A2PO, and GOLD.
- Update the PEFT continuation README example to use `peft_initialization.adapter_name=default`.

## [v2.3.0] - 2026-06-18

- Replace separate `SingleKVReward` and `MultiKVReward` wiring with one unified `KVReward` implementation for reward categories containing the configurable `reward.kv.category_token`.
- Replace `reward.weight.single_kv` and `reward.weight.multi_kv` with `reward.weight.kv`, and update GRPO, Async GRPO, A2PO, and SDPO launcher defaults to use the unified reward key.
- Add `reward.kv.*` controls for strict JSON parsing, required terminal stop token, invalid JSON reward, root-shape caps, stop-token caps, serialized-length caps, leaf-count caps, KV value/path weighting, table value/structure weighting, and match threshold.
- Change grounding reward category tokens to `bbox` for grounding bbox rewards and `evidence` for grounding selection rewards.
- Update README and reward documentation to describe unified KV reward routing, grounding category tokens, and reward activation rules.
- Refactor vLLM synchronization by extracting callback handling into shared sync helpers while preserving existing Qwen packed-MoE, sparse decoder MoE, and LoRA streaming sync paths.
- Normalize dense-to-MoE preprocessing and tokenizer merge formatting, and remove nested preprocessing helper structure from related verification utilities.
- Enforce Hydra config EOF discipline and ignore local `.jsonl` reference data artifacts under `references/`.
- Refresh `packages.txt` with current environment freeze additions for transitive parser packages.

## [v2.2.0] - 2026-06-17

- Add an Albumentations-backed VLM image augmentation backend selectable with `image_augmentation.backend=albumentations` while keeping `pil` as the default backend.
- Add extended Albumentations image degradation controls for resize, rotate, blur, noise, seasoning, coarse dropout, scan, color, HSV, RGB shift, JPEG, and weather transforms.
- Add pinned `albumentations==2.0.8` and `opencv-python-headless==4.13.0.92` dependencies to direct dependency files.
- Fix VLM preference dataset image conversion by passing the configured image conversion mode through DPO, GRPO, and KTO image augmentation paths.
- Refactor dataset path, image IO, train utility, reward, runtime, and preprocessing helpers to require explicit arguments instead of relying on implicit config access.
- Document Albumentations image augmentation usage, smoke-test overrides, grounding/bbox safety guidance, and vLLM max-model-length guidance for small Qwen3-VL GRPO smoke runs.
- Ignore local `.zip` archives so packaged local artifacts do not appear as repository changes.

## [v2.1.0] - 2026-06-16

- Add disabled-by-default auto KV stop-format reward shaping through `reward.auto_kv_stop_format` with configurable `stop_token`, blend weight, terminal reward, missing reward, and middle-or-multiple penalty values.
- Wire `reward.auto_kv_stop_format` into `SingleKVReward` and `MultiKVReward` so KV rewards can combine JSON correctness with terminal stop-token format compliance.
- Add strict KV JSON parsing for stop-format mode, including support for full JSON objects and fully fenced JSON blocks after a valid terminal stop token is stripped.
- Prevent malformed JSON or trailing-garbage KV outputs from receiving a high score from terminal stop-token formatting alone.
- Normalize base64 and converted VLM image sources to PNG data URIs so augmented DPO, GRPO, and KTO preference images remain serializable across dataset and downstream processing boundaries.
- Preserve existing path-based VLM image handling while passing already-normalized data URI inputs through unchanged.
- Filter chat template kwargs by actual template variables before calling tokenizer or processor chat template rendering, including `enable_thinking`.
- Finalize trainer chat template kwargs after data encoder setup so training arguments only receive kwargs supported by the active chat template.
- Refactor train dataset setup and data collator construction into shared setup helpers while preserving SFT dynamic padding behavior.
- Align test, large-test, vLLM test, and multi-turn vLLM test encoder/model setup order with the normalized data encoder flow.
- Document auto KV stop-format reward behavior and scoring constraints in the reward reference.

## [v2.0.0] - 2026-06-14

- Upgrade the training stack dependency pins to `transformers==5.11.0`, `trl==1.6.0`, `vllm==0.19.0`, `huggingface-hub==1.19.0`, `liger_kernel==0.8.0`, and related runtime packages.
- Switch KTO and SDPO configs to upstream TRL experimental trainers and configs, including `trl.experimental.kto.KTOTrainer`, `trl.experimental.kto.KTOConfig`, `trl.experimental.sdpo.SDPOTrainer`, and `trl.experimental.sdpo.SDPOConfig`.
- Remove the local SDPO trainer shim now superseded by the upstream TRL experimental SDPO trainer path.
- Add A2PO training support with method config, trainer config, training argument config, pipeline wiring, reward wiring, single-node launcher, and multi-node launchers.
- Add GOLD training support with method config, trainer config, training argument config, teacher-oriented defaults, single-node launcher, and multi-node launchers.
- Add KTO VLM dataset support with dataset config controls for `modality`, `decode_image_paths`, and `dataset_image`.
- Add VLM input support for `test_vllm` and `test_vllm_multi_turn` by passing resolved images through vLLM `multi_modal_data`.
- Update GRPO, SDPO, A2PO, GOLD, and related TRL 1.6 defaults for loss, vLLM server, colocated vLLM, importance sampling, and trainer-specific settings where applicable.
- Record expanded method and runtime metadata for the TRL 1.6 method surface, including A2PO, GOLD, upstream KTO, and upstream SDPO paths.
- Remove stale DPO and GRPO dataset image byte-loader paths in favor of normalized image-source handling.
- Document supported training methods, TRL 1.6 training options, SDPO teacher-server mode, A2PO/GOLD launchers, and VLM dataset image controls in the README.

## [v1.24.0] - 2026-06-12

- Add configurable experiment tracking backend support with W&B as the default backend and MLflow as an opt-in backend through `tracking={wandb,mlflow}`.
- Add `configs/tracking/wandb.yaml` and `configs/tracking/mlflow.yaml` with backend, trainer reporting, tracking URI, and artifact location settings.
- Compose tracking defaults into SFT, DPO, KTO, GKD, GRPO, SDPO, and Async GRPO configs.
- Route trainer `report_to` settings through `${tracking.report_to}` instead of hard-coding W&B in training argument configs.
- Add tracking utility APIs for train/eval initialization, table logging, alert handling, and backend finalization.
- Route train, test, large-model test, vLLM test, multi-turn test, and benchmark flows through the shared tracking backend helpers.
- Preserve W&B train run identity behavior by reusing the persisted `tracking_run_id` while keeping artifact `run_id` as checkpoint-local metadata.
- Add MLflow train/eval run support with artifact run-id tags and `tracking_metadata.json` mapping between artifact `run_id` and MLflow run UUID, with resume requiring existing tracking metadata.
- Add the pinned MLflow dependency to direct dependency files and ignore local tracking database files.
- Document tracking backend selection, MLflow storage defaults, and checkpoint artifact run-id mapping in README and usage guide updates.

## [v1.23.1] - 2026-06-12

- Align W&B train run identity with persisted `tracking_run_id` metadata so fresh artifact runs never reuse the checkpoint-local `run_id` as the W&B internal ID.
- Enable W&B resume handling for train runs with `resume="allow"` so interrupted-run resume reuses the persisted tracking identity from `tracking_metadata.json`.
- Keep the existing W&B display name based on `logging_name` while preserving `run_id` as the checkpoint-local artifact identifier.

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
