# MLflow Usage Guide

## Scope

This guide explains how to record training and evaluation runs from llm-fine-tune-hf in MLflow, inspect and compare results, resume interrupted runs, and diagnose common failures.

See `TRAINING_EVAL_CONTRACT.md` for the runtime contract and `README.md` and `USAGE_GUIDE.md` for the full execution options.

## Select a tracking profile

| Profile | Storage | Use case | Required configuration |
| --- | --- | --- | --- |
| `tracking=mlflow` | `${CONNECTED_DIR}/llm-fine-tune-hf/mlflow.db` and `${CONNECTED_DIR}/llm-fine-tune-hf/mlflow-artifacts` | Portable local tracking | A writable `CONNECTED_DIR` |
| `tracking=mlflow_server` | Remote MLflow server | Shared or centrally managed tracking | `MLFLOW_TRACKING_URI` and user credentials when required by the server |
| `tracking=wandb` | W&B | Explicit W&B tracking | W&B login and project access |

The portable default is `tracking=mlflow`. Select `tracking=mlflow_server` explicitly when a run must be recorded in a remote MLflow service. Local runs are not synchronized to a remote service automatically.

The remote profile does not set an Experiment artifact location from the client. It follows the artifact storage policy configured by the server. Do not override that location from the training command.

Tracking and lifecycle notifications are separate settings.

| Notification profile | Behavior |
| --- | --- |
| `notifications=slack` | Sends training start, training or evaluation completion and failure, and training interruption notifications through `SLACK_WEBHOOK_URL`. This is the default. |
| `notifications=wandb` | Sends alerts to the active W&B run. Use it with `tracking=wandb`. |
| `notifications=disabled` | Disables lifecycle notifications. |

If the Slack webhook is missing, preflight validation fails. If only a Slack request fails during execution, the original training or evaluation result is preserved. Diagnose notification and tracking failures separately.

## Prepare the environment

Use an environment installed according to the README. If the optional `myenv` name from the setup example was used:

```bash
conda activate myenv
python -c "import mlflow; print(mlflow.__version__)"
```

The project dependency configuration includes an MLflow compatibility override. Do not upgrade MLflow independently without revalidating the dependency set.

Copy the example environment file at the repository root and store real values only in `.env`.

```bash
cp .env.example .env
```

A remote MLflow run can use the following variables:

```dotenv
MLFLOW_TRACKING_URI=https://<mlflow-host>
MLFLOW_TRACKING_USERNAME=<username>
MLFLOW_TRACKING_PASSWORD=<password>
SLACK_WEBHOOK_URL=<webhook-url>
```

- Set the username and password only when the server uses MLflow Basic Auth.
- Do not use a shared administrator credential for routine training or evaluation.
- Do not put passwords, webhooks, or tokens in Hydra CLI overrides or shell arguments.
- Do not add `.env` or credential files to Git.
- Obtain the tracking URI and access policy from the operator of the selected MLflow service.

`main.py` loads `.env` from the repository root with `override=true`, so values in `.env` take precedence over process environment variables with the same names. Run commands from the repository root and confirm which environment file is active.

## Run with MLflow

Remote MLflow training:

```bash
python main.py mode=train tracking=mlflow_server
```

Disable notifications explicitly when Slack is unavailable:

```bash
python main.py mode=train tracking=mlflow_server notifications=disabled
```

Remote MLflow evaluation:

```bash
python main.py mode=test tracking=mlflow_server
python main.py mode=test_large tracking=mlflow_server
python main.py mode=test_vllm tracking=mlflow_server
python main.py mode=test_vllm_multi_turn tracking=mlflow_server
```

Local MLflow training uses the default profile:

```bash
python main.py mode=train
```

Run the local UI in another terminal. `CONNECTED_DIR` is a root directory, and the repository appends its package name when composing `connected_dir`.

```bash
connected_root="$(python -c "import dotenv; print(dotenv.dotenv_values(\".env\")[\"CONNECTED_DIR\"])")"
mlflow ui --backend-store-uri "sqlite:///${connected_root}/llm-fine-tune-hf/mlflow.db"
```

Do not use one local SQLite database as a shared backend across multiple hosts.

Select both W&B profiles when using W&B tracking and W&B alerts:

```bash
python main.py mode=train tracking=wandb notifications=wandb
```

## Preflight checklist

1. Record the Git revision and working tree state.

   ```bash
   git rev-parse --short HEAD
   git status --short
   ```

2. Confirm the model, model revision, dataset, method, modality, and strategy.
3. Confirm the selected local or remote tracking profile and URI.
4. Verify Experiment read and write access before starting a remote run.
5. Confirm that output paths and visible GPUs do not collide with another run.
6. Distinguish a fresh run from a resume of an interrupted run.
7. Confirm that samples, completion text, and images are allowed in the selected artifact store.

## Experiment and run identity

| Repository value | MLflow use | Default composition |
| --- | --- | --- |
| `project_name` | Experiment name | `${model_name}-${effective_dataset_name}-${mode}` |
| `logging_name` | Training run display name | `${save_detail}-lr${lr}` |
| `model_detail` | Evaluation run display name | Composed by the model config |
| `run_id` | Checkpoint and file artifact directory identifier | `run-0001`, `run-0002`, and so on |
| `tracking_run_id` | MLflow run identifier | Stored in `${output_dir}/tracking_metadata.json` |

`run_id` and `tracking_run_id` have different scopes.

- The `artifact_run_id` tag contains the output namespace leaf ID and is not globally unique. Use it together with `output_dir` when disambiguation is required.
- Use `artifact_run_id` or `output_dir` tags to locate files.
- Use `tracking_run_id` for the MLflow API, run URL, and resume identity.
- Display names may be duplicated. Include `tracking_run_id` when sharing results and include `artifact_run_id` for training runs.
- Keep the default Experiment namespace when model, dataset, or mode changes.

## Automatically recorded data

Training and evaluation runs record these tags:

- `mode`
- `package_name` and `package_version`
- `project_name`
- `fine_tune_method`
- `modality`
- `dataset_name`
- `model_type` and `model_revision`
- `strategy`

Training also records `artifact_run_id`, `output_dir`, `logging_name`, and `resume_training`. `git_revision` is added from the manifest during terminal metadata logging and may not be visible while the run is active. Evaluation adds `model_detail`.

Transformers and TRL report loss, learning rate, evaluation metrics, and method-specific scalar metrics through `report_to=mlflow`. System metrics use a 10-second sampling interval by default, and GPU collection is limited to devices selected by `CUDA_VISIBLE_DEVICES`. Metric semantics and reporting intervals vary by method. Do not substitute a fabricated zero for a metric that was not produced.

Before a training run receives its terminal status, available files from this list are validated and uploaded under the MLflow `metadata/` artifact path:

- `run_manifest.json`: lifecycle, last stage, Git revision, observed inputs, runtime facts, and artifact references
- `resolved_config.yaml`: fully resolved configuration with secrets redacted
- `training_args.json`: effective trainer arguments
- `tracking_metadata.json`: backend and `tracking_run_id`

All candidate files are checked before upload. If an unredacted secret is detected, the complete metadata upload fails closed. Do not bypass this validation with a manual upload.

Evaluation modes log the result DataFrame as the `test_results.json` MLflow table artifact. Canonical results and companion manifests are also stored under `${CONNECTED_DIR}/llm-fine-tune-hf/tests/...` according to `TRAINING_EVAL_CONTRACT.md`.

When GRPO uses `log_completions=true`, completion tables are stored under `completions/tables/`. If `log_multimodal=true` is also enabled, referenced images are stored under `completions/images/`. Evaluation output, prompts, completions, and images may contain source data, so the person running the experiment must confirm that central storage is permitted.

## Monitor an active run

1. Open the Experiment named by `project_name`.
2. For training, confirm that the display name and `artifact_run_id` match the local output directory.
3. Confirm that the status is `RUNNING`.
4. Check that trainer and system metrics update at the expected intervals.
5. For a single process, confirm that the number of GPU metric devices matches the visible GPU count. Apply the distributed guidance below to multi-process runs.
6. For training, confirm that no duplicate run points to the same `output_dir`.

If training is active but the UI does not update, do not immediately start another run. Inspect the process log, network and authentication state, and the timestamp of the last metric first.

## Search and compare runs

MLflow UI filter conditions can be combined with `AND`.

Find completed SFT runs for the default dataset:

```text
attributes.status = 'FINISHED' AND tags.fine_tune_method = 'sft' AND tags.dataset_name = 'tulu'
```

Find a specific training artifact:

```text
tags.artifact_run_id = 'run-0001' AND tags.output_dir = '<resolved-output-dir>'
```

Find runs using the same model and Git revision:

```text
tags.model_revision = 'main' AND tags.git_revision = '<git-revision>'
```

Before comparing runs, align the dataset source, version and split, model ID and revision, method, modality, metric definition, and evaluation conditions. Confirm whether differences in batch size, seed, strategy, and distributed topology are intended experimental variables.

Line chart smoothing changes only the UI display. It does not modify stored metrics or exported values. MLflow 3.13.0 provides a workspace-wide `Line smoothing` control under `Configure charts`, and each chart can select `Use workspace settings` or `Custom`. The 0-100 control converts its value to an EMA historical coefficient using `value / 101`, so `100` produces approximately `0.9901` and is the closest setting to EMA `0.99`. W&B and MLflow may still render different curves because their smoothing implementations are not identical. Apply the same workspace or custom value to the loss, evaluation loss, mean token accuracy, reward, and GPU metric charts being compared. Lower or disable smoothing when inspecting raw step-level variation.

When sharing a result, include the Experiment, display name, `tracking_run_id`, Git revision, important config differences, and inspected metrics. Include `artifact_run_id` for a training result.

## Resume or rerun

```bash
resume_from_checkpoint="/path/model-is_sft=True-loss_type=sapo/run-0001/checkpoint-940"
backslash="\\"
escaped_checkpoint="${resume_from_checkpoint//=/${backslash}=}"

hydra_overrides=(
  "mode=train"
  "tracking=mlflow_server"
  "resume_training=true"
  "resume_from_checkpoint=${escaped_checkpoint}"
)

python main.py "${hydra_overrides[@]}"
```

The first `=` in `resume_from_checkpoint=...` separates the Hydra key from its value. Every `=` inside the checkpoint path must be escaped as `\=`. Before launching, confirm that the checkpoint directory contains `trainer_state.json` and the optimizer and scheduler state required by the selected training strategy. When continuing an existing log, append with `>>` rather than overwrite it with `>`. After launch, inspect the effective process arguments and confirm that training advances beyond checkpoint step `940` without `OverrideParseException`, traceback, or OOM.

- Resume reopens the MLflow run identified by `tracking_run_id` in the existing `tracking_metadata.json`.
- If the identity file is missing or the ID is empty, resume fails instead of creating a new run.
- `tracking_metadata.json` stores only `backend=mlflow`, not the local or server profile or URI. Check the previous `tracking.tracking_uri` in `resolved_config.yaml` and do not switch between local and remote profiles or between remote endpoints during resume.
- Resume fails when the stored backend and selected backend differ.
- Immutable MLflow parameters are not logged again during resume.
- A fresh run fails instead of overwriting an existing `tracking_metadata.json`.
- Do not create an identity file manually or copy one from another run.

If the checkpoint or tracking identity cannot be trusted, allocate a new artifact run and record its relationship to the previous run separately.

## Terminal states and abnormal termination

| Situation | MLflow status | Training manifest status |
| --- | --- | --- |
| Normal completion | `FINISHED` | `completed` |
| Ordinary exception | `FAILED` | `failed` |
| `KeyboardInterrupt` or `SystemExit` | `KILLED` | `interrupted` |

The manifest status applies only to training runs. Use the MLflow status for the evaluation lifecycle.

An additional tracking finalization or notification failure does not replace the original pipeline exception.

`SIGKILL`, host reboot, and GPU driver reset can prevent finalization and leave a run in `RUNNING` state.

1. Confirm through the host or scheduler that the process has stopped.
2. Compare `tracking_run_id` and host logs. For training, also inspect `artifact_run_id` and the last status and stage in `run_manifest.json`.
3. Resume with the same tracking identity when a valid checkpoint exists.
4. If the run will not be resumed, confirm the original tracking URI and use an authorized MLflow API credential to mark it `KILLED` and record the cause.

   ```python
   import dotenv

   dotenv.load_dotenv(
       dotenv_path=".env",
       override=True,
   )

   from mlflow import MlflowClient

   client = MlflowClient(
       tracking_uri="<tracking-uri-used-by-run>",
   )
   client.set_tag(
       run_id="<tracking-run-id>",
       key="termination_reason",
       value="<non-sensitive-cause>",
   )
   client.set_terminated(
       run_id="<tracking-run-id>",
       status="KILLED",
   )
   ```

   Keep `termination_reason` non-sensitive. Do not include credentials, private URLs, or confidential payloads.

5. Do not start a new run in the same output directory before identifying the cause.

## Distributed training

- Only rank 0 creates and terminates the tracking run.
- If a non-rank-0 tracking run appears, inspect launcher values for `RANK`, `WORLD_SIZE`, and the distributed config.
- Rank 0 allocates the shared output `run_id`; other ranks wait for the allocation metadata.
- GPU metrics may cover only the devices visible to the rank 0 process rather than every GPU in the cluster.
- Compare node count, process count, machine rank, effective batch size, and DataLoader runtime from the manifest.

## Slack notifications

Messages include status, project, run name, `artifact_run_id`, `tracking_run_id`, backend, method, model, dataset, and output directory. Evaluation runs may have `artifact_run_id=None`. A run URL is included for an HTTP or HTTPS MLflow server.

- If the webhook is empty, check `.env` or select `notifications=disabled`.
- For timeout, DNS, proxy, or firewall errors, inspect outbound network policy for the active environment.
- If only Slack is missing and the run finishes correctly, do not classify it as a tracking failure.
- If both MLflow and Slack fail, diagnose network, DNS, and host access separately for each endpoint.

Never put the webhook URL in logs, screenshots, issues, commits, tags, parameters, or artifacts.

## Troubleshooting

### `tracking.tracking_uri is required`

- Confirm `tracking=mlflow_server` and `MLFLOW_TRACKING_URI` in `.env`.
- Check the variable name, surrounding whitespace, and current working directory.
- Obtain the URI from the service operator instead of guessing it.

### Authentication or permission failure

- Confirm that the user credential and URI belong to the same environment.
- Request only the Experiment permissions required for the run.
- Do not bypass access control with an administrator or another user's credential.

### A run exists but has no metrics

- Confirm from the process log that the trainer entered a training step.
- Check `tracking.backend=mlflow` and the reporting backend.
- Inspect the manifest for setup or preflight failure.
- Confirm that no previous process is still active before deciding to rerun.

### GPU metrics are missing or the count differs

- Check `tracking.system_metrics.enabled=true` and `CUDA_VISIBLE_DEVICES`.
- Confirm that host NVML can resolve the configured GPU UUIDs or indices.
- Check GPU and NVML visibility inside the container and the rank 0 visible device set.

### Metadata upload fails

- Inspect metadata files created in the output directory.
- If terminal metadata logging fails after successful training, the command exits with an error and the MLflow status becomes `FAILED`. The local manifest may already be `completed`, so inspect both states.
- Do not change the run to `FINISHED` before resolving the cause.
- If secret detection fails, do not share or manually upload the file.
- Check whether credentials were placed in a nonstandard config key or free-form string.

### Resume is rejected

- Check `resume_training`, the checkpoint path, and `tracking_metadata.json`.
- Compare the previous `tracking.tracking_uri` in `resolved_config.yaml` with the current URI.
- Confirm that the stored and selected backends are identical.
- Confirm that a fresh output directory was not confused with a resume directory.

### The evaluation table is missing

- Confirm that evaluation reached the result storage stage.
- Inspect the MLflow `test_results.json`, canonical result, and companion manifest together.
- For multi-rank evaluation, confirm that rank 0 completed gathering and logging.

## Run checklist

### Before starting

- [ ] Git revision and working tree state are recorded.
- [ ] Model, revision, dataset, method, modality, and strategy are confirmed.
- [ ] The intended local or remote tracking profile is selected.
- [ ] User credentials and required Experiment permissions are available.
- [ ] Fresh run and resume behavior are distinguished.
- [ ] Output and GPU namespaces do not collide, and centrally stored data is permitted.

### While running

- [ ] Experiment and `tracking_run_id` are correct, and `artifact_run_id` is also checked for training.
- [ ] No duplicate run exists.
- [ ] Trainer metrics, system metrics, and visible GPU scope are correct.
- [ ] Notification failure and tracking failure are diagnosed separately.

### After completion

- [ ] MLflow and manifest states match the actual result.
- [ ] Important metrics, tags, and Git revision are present.
- [ ] Required metadata, evaluation tables, and completion artifacts are present.
- [ ] Failed or interrupted runs have a clear resume or termination decision.
- [ ] Shared results include `tracking_run_id` and include `artifact_run_id` for training.

## Related documentation and configuration

- `README.md`: installation, configuration, and execution examples
- `USAGE_GUIDE.md`: training and evaluation runbook
- `TRAINING_EVAL_CONTRACT.md`: tracking lifecycle and output contract
- `.env.example`: supported environment variables
- `configs/tracking/mlflow.yaml`: local MLflow profile
- `configs/tracking/mlflow_server.yaml`: remote MLflow profile
- `configs/notifications/slack.yaml`: Slack notification profile
