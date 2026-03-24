# Changelog

All notable changes to this repository are documented in this file.

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
