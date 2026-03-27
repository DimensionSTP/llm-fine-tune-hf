# Changelog

All notable changes to this repository are documented in this file.

## [v1.9.0] - 2026-03-27

- Add configurable negative reward penalties for GRPO match and code-execution rewards.
- Wire negative-penalty settings through `configs/grpo.yaml` and `configs/reward/manager.yaml`.
- Include active negative-penalty settings in Hydra artifact suffix generation for clearer experiment tracking.
- Add Mixtral-style dense-to-MoE preprocessing and verification utilities for non-Qwen model families.
- Add Ministral3 text-backbone extraction and verification preprocessing entrypoints.
- Extend dense-LoRA-to-MoE merge and verification logic to support packed-expert MoE layouts.
- Export the new preprocessing entrypoints from `src.preprocessing`.

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
