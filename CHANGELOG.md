# Changelog

All notable changes to this repository are documented in this file.

## [v1.9.2] - 2026-04-06

- Sync `packages.txt`, `requirements.txt`, and `pyproject.toml` with the current `joshpp` runtime freeze after compatibility-driven dependency adjustments.
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
