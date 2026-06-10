# Rewards

This document summarizes the reward implementations in `src/utils/rewards.py`,
the category keywords they expect, and the configurable options exposed by
`configs/grpo.yaml` and `configs/reward/*.yaml`.

## Common Configuration (applies to all rewards)

These fields are wired in `configs/reward/manager.yaml` for every reward:

- `is_answer_tag`: when true, `extract_answer_from_generation()` pulls content
  between `answer_start_token` and `answer_end_token` instead of other patterns.
- `think_start_token`, `think_end_token`: used by `ThinkFormatReward` and as a
  fallback extraction boundary for other rewards.
- `answer_start_token`, `answer_end_token`: used to extract answer content when
  `is_answer_tag` is enabled.
- `eos_token`: used as a fallback answer boundary.
- `weight`: scales the computed reward; rewards with `weight <= 0` are skipped
  by `RewardManager`.

## Reward Catalog

### ThinkFormatReward

- Purpose: Enforce presence of a thinking block when thinking is enabled.
- Logic: If `is_enable_thinking` is true, reward is `1.0` when a single
  `<think>...</think>` block is detected; otherwise `0.0`. Returns `None` for
  all samples when `is_enable_thinking` is false.
- reward_categories: not used (applies to all samples).
- Extra options:
  - `is_enable_thinking`: global toggle; when false, this reward returns `None`.

### AnswerFormatReward

- Purpose: Enforce presence of an answer tag block.
- Logic: Reward is `1.0` when an `<answer>...</answer>` block is detected;
  otherwise `0.0`.
- reward_categories: not used (applies to all samples).
- Extra options: none.

### MatchReward

- Purpose: Exact or near-exact match for short answers (math / choice tasks).
- Logic: Extracts answer text, tries exact match, then wrapper-stripped match,
  normalized match, choice letter match, and number match. Returns `1.0` on
  any match, otherwise `0.0`.
- reward_categories: `math`, `choice`.
- Extra options: none.

### CodeExecutionReward

- Purpose: Compare executable Python outputs between prediction and solution.
- Logic: Extracts a fenced ` ```python ... ``` ` block from the prediction,
  executes it with a timeout, and compares stdout to the solution's executed
  stdout. Reward is `1.0` if outputs match, `0.5` if both run but outputs differ,
  otherwise `0.0`. Returns `None` if the solution cannot be executed.
- reward_categories: `code`.
- Extra options:
  - `timeout`: maximum execution time (seconds).

### RougeReward

- Purpose: ROUGE-F1 similarity for open-form text.
- Logic: Extracts answer text, normalizes wrappers, and computes ROUGE-1/2/L
  F1 against the solution.
- reward_categories: `rouge`.
- Extra options:
  - `rouge_type`: one of `1`, `2`, `l`.

### EquationReward

- Purpose: Reward algebraic equation correctness under number constraints.
- Logic: Parses numbers used in the predicted equation and checks they match
  the provided list. Validates allowed operators and evaluates the equation;
  returns `1.0` if the computed value matches the target, otherwise `0.0`.
- reward_categories: `equation`.
- Extra options:
  - `equation_target_column_name`: key in the solution dict for target value.
  - `equation_numbers_column_name`: key in the solution dict for allowed numbers.

### RetrievalHitReward

- Purpose: Reward query rewriting that improves retrieval hits/ranks.
- Logic: Embeds the original query and the rewritten query (prediction), runs
  retrieval, and compares hit positions against ground-truth candidates. Gives
  a shaped score based on rank improvement and stage bonus/drop.
- reward_categories: any category containing the `retrieval` token
  (e.g., `retrieval`, `retrieval_hit`, `retrieval_ndcg`).
- Extra options:
  - `shaping_weight`: scaling for rank-improvement shaping.
  - `rank_margin`: ignore small rank changes within this margin.
  - `stages`: rank-stage bonus/drop settings.
  - `reward_database` config (`configs/reward/database.yaml`):
    - `data_path`, `embedding_model_detail`, `indices_name`, `items_name`,
      `items_format`, `dim`, `retrieval_top_k`, `distance_column_name`,
      `candidate_column_name`.
  - `reward_embedding` config (`configs/reward/embedding.yaml`):
    - `model_path`, `model_base`, `model_ft_data`, `model_default_upload_user`,
      `num_gpus`, `seed`, `gpu_memory_utilization`, `max_length`,
      `instruction_length`, `instruction`, `device_id`, `master_addr`,
      `master_port`, `nccl_socket_ifname`, `nccl_ib_disable`,
      `preserved_env_keys`, `isolated_env_keys`.
    - `preserved_env_keys` are restored after reward embedding vLLM
      initialization so colocated training keeps the original distributed
      environment.
    - `isolated_env_keys` are temporarily removed before reward embedding vLLM
      initialization to avoid inheriting trainer rank state.

### RetrievalnDCGReward

- Purpose: Reward retrieval quality of rewritten queries using weighted nDCG.
- Logic: Computes weighted nDCG over configured cutoffs (`top_ks`) using
  power-law weights with configurable cutoff emphasis. Supports two reward modes:
  - `relative`: headroom-normalized delta from original to rewritten query.
  - `absolute`: rewritten-query nDCG only.
- reward_categories: any category containing the `retrieval` token
  (e.g., `retrieval`, `retrieval_hit`, `retrieval_ndcg`).
- Extra options:
  - `reward_mode`: `relative` (default) or `absolute`.
  - `top_ks`: nDCG cutoff list (strictly increasing, each <= retrieval top-k).
  - `alpha`: power exponent for cutoff weights.
  - `weighting_mode`: `small_k` for smaller-cutoff emphasis (`k^-alpha`),
    `large_k` for larger-cutoff emphasis (`k^alpha`). Current GRPO default is
    `large_k`.
  - `epsilon`: numerical stabilizer (used in `relative` mode).
  - `reward_database` / `reward_embedding`: same retrieval backends as
    `RetrievalHitReward`.

### SingleKVReward

- Purpose: Single key-value extraction reward for JSON outputs.
- Logic: Parses JSON from prediction and solution. If JSON contains a `tables`
  section, normalizes table `rows` from either list or dict containers and
  scores per-cell matches. Otherwise compares the last leaf value with
  normalized matching. Returns `1.0` on match, else `json_parse_weight`. Returns
  `0.0` on parse failure.
- reward_categories: any category containing the `kv` token
  (e.g., `single_kv`, `vlm_single_kv`).
- Extra options:
  - `json_parse_weight`: base score for valid JSON even if value mismatches.

### MultiKVReward

- Purpose: Multi key-value extraction reward for JSON outputs.
- Logic: Parses JSON, compares all leaf values (excluding `tables`) and table
  cells after list/dict `rows` normalization. Computes accuracy over total items and returns
  `json_parse_weight + (1 - json_parse_weight) * accuracy`. Returns `0.0` on
  parse failure.
- reward_categories: any category containing the `kv` token
  (e.g., `multi_kv`, `vlm_multi_kv`).
- Extra options:
  - `json_parse_weight`: base score for valid JSON even if accuracy is low.

### GroundingBBoxReward

- Purpose: Ground a field answer to page-level bounding boxes and reject
  missing evidence when the target is not found.
- Logic: Parses JSON from the extracted answer. For labels whose
  `grounding_status` is `found`, scores schema validity, page match, best-box
  IoU, IoU-threshold bonuses, and center-in-ground-truth bonus. Penalizes overly
  large predicted boxes and overlap with `hard_negative_evidence`. For labels
  whose `grounding_status` is not `found`, a prediction with non-`found`
  `grounding_status` and empty `evidence_occurrences` receives `max_reward`.
  Invalid JSON returns `0.0`; samples without matching category return `None`.
- reward_categories: any category containing
  `reward.grounding_bbox.category_token` (default: `ground`).
- Expected label schema in `solution`:
  - `grounding_status`: `found` or a non-`found` status.
  - `coord_system`: optional coordinate-system string checked against predicted
    boxes when present.
  - `positive_occurrences`: required for `found` labels. Each occurrence should
    provide `page` plus either `fragments[].bbox` or `envelope_bbox`.
  - `hard_negative_evidence`: optional list of boxes with `page` and `bbox`.
- Expected prediction schema:
  - `field_path`: field identifier string.
  - `grounding_status`: predicted grounding status.
  - `evidence_occurrences`: list of predicted occurrences. Each occurrence uses
    the same `page`, `fragments[].bbox`, or `envelope_bbox` shape as labels.
- Extra options:
  - `category_token`: category token that activates this reward.
  - `schema_keys`, `status_values`: narrow aliases for semantically equivalent
    schema/status keys. Defaults preserve the existing `field_path`,
    `grounding_status`, `evidence_occurrences`, and `positive_occurrences`
    schema.
  - `format_reward`, `schema_reward`, `page_reward`: base shaping components.
  - `iou_weight`, `iou_05_threshold`, `iou_05_bonus`, `iou_07_threshold`,
    `iou_07_bonus`, `center_in_gt_bonus`: positive grounding match shaping.
  - `large_box_area_threshold`, `large_box_penalty`: large-box penalty.
  - `hard_negative_iou_threshold`, `hard_negative_overlap_penalty`: hard
    negative overlap penalty.
  - `positive_duplicate_iou_threshold`: filters hard negatives that duplicate a
    positive target.
  - `min_reward`, `max_reward`: final clipping bounds.

### GroundingSelectionReward

- Purpose: Select the correct grounding candidate ids when the task provides a
  candidate list instead of requiring generated boxes.
- Logic: Parses JSON from the extracted answer. The prediction and solution are
  top-level objects with a `grounding` list. Items are matched by `target_id`,
  and each gold target must have a prediction item. The reward compares each
  item's `selected_ids` with the gold `selected_ids`, supports empty selections,
  penalizes wrong, over-selected, duplicate, missing, and extra target ids, and
  does not crash on malformed JSON. It does not use `grounding_status`; value
  targets are expected to provide evidence candidate ids through `selected_ids`.
- reward_categories: any category containing
  `reward.grounding_selection.category_token` (default: `grounding`).
- Expected label schema in `solution`:
  - top-level `grounding`: list of target selection items.
  - item `target_id`: unique target id.
  - item `selected_ids`: correct candidate ids for that target.
- Expected prediction schema:
  - top-level `grounding`: list of predicted selection items.
  - item `target_id`: target id to match against the solution.
  - item `selected_ids`: predicted candidate ids. The
    `selected_candidate_ids` alias is also supported by default.
- Extra options:
  - `category_token`: category token that activates this reward.
  - `schema_keys`: narrow aliases for `items`, `target_id`, and
    `selected_ids`.
  - `format_reward`, `schema_reward`, `exact_match_reward`,
    `partial_match_weight`: positive shaping components.
  - `over_selection_penalty`, `wrong_selection_penalty`: selection penalties.
  - `min_reward`, `max_reward`: final clipping bounds.

## reward_categories Quick Reference

Current matching rules in `src/utils/rewards.py`:

- `math`
- `choice`
- `code`
- `rouge`
- `equation`
- Retrieval-family rewards (`RetrievalHitReward`, `RetrievalnDCGReward`) run
  when category contains token `retrieval`.
  Examples: `retrieval`, `retrieval_hit`, `retrieval_ndcg`.
- KV-family rewards (`SingleKVReward`, `MultiKVReward`) run when category
  contains token `kv`.
  Examples: `single_kv`, `multi_kv`, `vlm_single_kv`, `vlm_multi_kv`.
- Grounding rewards run when category contains their configured category token:
  `reward.grounding_bbox.category_token` or
  `reward.grounding_selection.category_token` (default: `grounding` for
  selection).
