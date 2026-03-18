# Contributing Guide

## Scope

This repository is maintained for team collaboration and reproducible execution.
Changes should keep behavior clear, traceable, and compatible with existing workflows.

## Workflow

1. Open an issue (or include clear context in the PR) describing:
- objective
- impacted files
- expected downstream effect

2. Keep changes minimal and purpose-focused:
- one logical purpose per commit
- avoid mixing unrelated edits

3. Update dependent docs and configs when contracts or paths change.

## Commit Rules

- Use commit message format: `<type>: <concise purpose>`
- Recommended types: `add`, `fix`, `feat`, `refactor`, `rename`, `delete`, `docs`, `chore`
- Prefer split commits by logical behavior, not by proximity

## Pull Request Checklist

- [ ] Purpose and scope are explicit
- [ ] Affected files are listed
- [ ] Output/contract impact is described
- [ ] Backward compatibility or migration note is added (if needed)
- [ ] README / usage references are updated (if needed)
- [ ] Local checks or CI checks passed

## Release and Tags

- Update `CHANGELOG.md` before creating a release tag.
- Use semantic-style tags (recommended): `vMAJOR.MINOR.PATCH`.
- For docs/chore-only updates, increment `PATCH` unless breaking changes are introduced.
