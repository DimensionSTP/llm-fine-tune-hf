from typing import Dict, List, Tuple, Set, Optional, Any
from difflib import SequenceMatcher
import json
import math
import re
import unicodedata

from omegaconf import ListConfig

from .base import BaseReward


class KVReward(BaseReward):
    def __init__(
        self,
        is_answer_tag: bool,
        think_start_token: str,
        think_end_token: str,
        answer_start_token: str,
        answer_end_token: str,
        eos_token: str,
        extraction_profile: str,
        weight: float,
        category_token: str,
        strict_json: bool,
        required_stop_token: str,
        invalid_json_reward: float,
        root_mismatch_cap: float,
        missing_stop_cap: float,
        middle_or_multiple_stop_cap: float,
        trailing_text_cap: float,
        max_serialized_length_ratio: float,
        length_ratio_cap: float,
        max_leaf_count_ratio: float,
        leaf_count_ratio_cap: float,
        kv_value_weight: float,
        kv_path_weight: float,
        table_value_weight: float,
        table_structure_weight: float,
        table_name_mismatch_cap: float,
        table_column_mismatch_cap: float,
        table_row_count_mismatch_cap: float,
        kv_top_level_path_mismatch_cap: float,
        extra_leaf_ratio_threshold: float,
        extra_leaf_cap: float,
        missing_leaf_ratio_threshold: float,
        missing_leaf_cap: float,
        duplicate_value_cap: float,
        duplicate_path_cap: float,
        duplicate_table_row_cap: float,
        use_weighted_fuzzy_f1: bool,
        weighted_fuzzy_match_threshold: float,
        normalized_match_cap: float,
        coarse_match_cap: float,
        non_exact_value_cap: float,
        empty_value_mismatch_cap: float,
        numeric_punctuation_mismatch_cap: float,
        score_threshold: float,
        allowed_sibling_keys: List[str],
    ) -> None:
        BaseReward.__init__(
            self,
            is_answer_tag=is_answer_tag,
            think_start_token=think_start_token,
            think_end_token=think_end_token,
            answer_start_token=answer_start_token,
            answer_end_token=answer_end_token,
            eos_token=eos_token,
            extraction_profile=extraction_profile,
            weight=weight,
        )
        self.category_token = category_token
        self.strict_json = strict_json
        self.required_stop_token = required_stop_token
        self.invalid_json_reward = invalid_json_reward
        self.root_mismatch_cap = root_mismatch_cap
        self.missing_stop_cap = missing_stop_cap
        self.middle_or_multiple_stop_cap = middle_or_multiple_stop_cap
        self.trailing_text_cap = trailing_text_cap
        self.max_serialized_length_ratio = max_serialized_length_ratio
        self.length_ratio_cap = length_ratio_cap
        self.max_leaf_count_ratio = max_leaf_count_ratio
        self.leaf_count_ratio_cap = leaf_count_ratio_cap
        self.kv_value_weight = kv_value_weight
        self.kv_path_weight = kv_path_weight
        self.table_value_weight = table_value_weight
        self.table_structure_weight = table_structure_weight
        self.table_name_mismatch_cap = table_name_mismatch_cap
        self.table_column_mismatch_cap = table_column_mismatch_cap
        self.table_row_count_mismatch_cap = table_row_count_mismatch_cap
        self.kv_top_level_path_mismatch_cap = kv_top_level_path_mismatch_cap
        self.extra_leaf_ratio_threshold = extra_leaf_ratio_threshold
        self.extra_leaf_cap = extra_leaf_cap
        self.missing_leaf_ratio_threshold = missing_leaf_ratio_threshold
        self.missing_leaf_cap = missing_leaf_cap
        self.duplicate_value_cap = duplicate_value_cap
        self.duplicate_path_cap = duplicate_path_cap
        self.duplicate_table_row_cap = duplicate_table_row_cap
        self.use_weighted_fuzzy_f1 = use_weighted_fuzzy_f1
        self.weighted_fuzzy_match_threshold = weighted_fuzzy_match_threshold
        self.normalized_match_cap = normalized_match_cap
        self.coarse_match_cap = coarse_match_cap
        self.non_exact_value_cap = non_exact_value_cap
        self.empty_value_mismatch_cap = empty_value_mismatch_cap
        self.numeric_punctuation_mismatch_cap = numeric_punctuation_mismatch_cap
        self.score_threshold = score_threshold
        self.allowed_sibling_keys = self._normalize_allowed_sibling_keys(
            allowed_sibling_keys=allowed_sibling_keys,
        )
        self._validate_kv_reward_config()

    @property
    def name(self) -> str:
        return "kv_reward"

    def compute(
        self,
        completions: List[List[Dict[str, str]]],
        solution: List[str],
        reward_categories: List[str],
        **kwargs,
    ) -> List[Optional[float]]:
        rewards = []
        contents = self.get_contents_from_completions(completions=completions)
        for content, sol, category in zip(contents, solution, reward_categories):
            if not self.has_category_token(
                category=category,
                token=self.category_token,
            ):
                rewards.append(None)
                continue

            if not sol:
                rewards.append(None)
                continue

            answer_text, format_cap = self._extract_json_text_and_format_cap(
                generation=content,
            )
            pred_json = self._parse_prediction_json(text=answer_text)
            if pred_json is None:
                rewards.append(self.invalid_json_reward)
                continue

            gt_json = self._parse_solution_json(text=sol)
            if gt_json is None:
                rewards.append(None)
                continue

            root_name = self._resolve_gt_root_name(gt_json=gt_json)
            if root_name is None:
                rewards.append(None)
                continue

            root_cap = self._compute_root_cap(
                pred_json=pred_json,
                root_name=root_name,
            )
            length_cap = self._compute_length_cap(
                pred_json=pred_json,
                gt_json=gt_json,
                root_name=root_name,
            )
            content_score = self._compute_root_score(
                pred_json=pred_json,
                gt_json=gt_json,
                root_name=root_name,
            )
            structure_cap = self._compute_structure_cap(
                pred_json=pred_json,
                gt_json=gt_json,
                root_name=root_name,
            )
            reward = min(
                content_score,
                format_cap,
                root_cap,
                length_cap,
                structure_cap,
            )
            rewards.append(self._clip_unit_score(score=reward))

        return rewards

    def _validate_kv_reward_config(
        self,
    ) -> None:
        if not isinstance(self.category_token, str) or not self.category_token.strip():
            raise ValueError("reward.kv.category_token must be a non-empty string")
        if not isinstance(self.strict_json, bool):
            raise ValueError("reward.kv.strict_json must be a bool")
        if not isinstance(self.use_weighted_fuzzy_f1, bool):
            raise ValueError("reward.kv.use_weighted_fuzzy_f1 must be a bool")
        if not all(
            isinstance(key, str) and key and key == key.strip()
            for key in self.allowed_sibling_keys
        ):
            raise ValueError(
                "reward.kv.allowed_sibling_keys must contain stripped non-empty strings"
            )
        if (
            not isinstance(self.required_stop_token, str)
            or not self.required_stop_token
        ):
            raise ValueError("reward.kv.required_stop_token must be a non-empty string")
        for name, value in [
            ("invalid_json_reward", self.invalid_json_reward),
            ("root_mismatch_cap", self.root_mismatch_cap),
            ("missing_stop_cap", self.missing_stop_cap),
            ("middle_or_multiple_stop_cap", self.middle_or_multiple_stop_cap),
            ("trailing_text_cap", self.trailing_text_cap),
            ("length_ratio_cap", self.length_ratio_cap),
            ("leaf_count_ratio_cap", self.leaf_count_ratio_cap),
            ("kv_value_weight", self.kv_value_weight),
            ("kv_path_weight", self.kv_path_weight),
            ("table_value_weight", self.table_value_weight),
            ("table_structure_weight", self.table_structure_weight),
            ("table_name_mismatch_cap", self.table_name_mismatch_cap),
            ("table_column_mismatch_cap", self.table_column_mismatch_cap),
            ("table_row_count_mismatch_cap", self.table_row_count_mismatch_cap),
            ("kv_top_level_path_mismatch_cap", self.kv_top_level_path_mismatch_cap),
            ("extra_leaf_ratio_threshold", self.extra_leaf_ratio_threshold),
            ("extra_leaf_cap", self.extra_leaf_cap),
            ("missing_leaf_ratio_threshold", self.missing_leaf_ratio_threshold),
            ("missing_leaf_cap", self.missing_leaf_cap),
            ("duplicate_value_cap", self.duplicate_value_cap),
            ("duplicate_path_cap", self.duplicate_path_cap),
            ("duplicate_table_row_cap", self.duplicate_table_row_cap),
            ("weighted_fuzzy_match_threshold", self.weighted_fuzzy_match_threshold),
            ("normalized_match_cap", self.normalized_match_cap),
            ("coarse_match_cap", self.coarse_match_cap),
            ("non_exact_value_cap", self.non_exact_value_cap),
            ("empty_value_mismatch_cap", self.empty_value_mismatch_cap),
            (
                "numeric_punctuation_mismatch_cap",
                self.numeric_punctuation_mismatch_cap,
            ),
            ("score_threshold", self.score_threshold),
        ]:
            self._validate_unit_number(
                name=name,
                value=value,
            )
        for name, value in [
            ("max_serialized_length_ratio", self.max_serialized_length_ratio),
            ("max_leaf_count_ratio", self.max_leaf_count_ratio),
        ]:
            self._validate_positive_number(
                name=name,
                value=value,
            )
        self._validate_weight_sum(
            first_name="kv_value_weight",
            first_value=self.kv_value_weight,
            second_name="kv_path_weight",
            second_value=self.kv_path_weight,
        )
        self._validate_weight_sum(
            first_name="table_value_weight",
            first_value=self.table_value_weight,
            second_name="table_structure_weight",
            second_value=self.table_structure_weight,
        )

    @staticmethod
    def _validate_unit_number(
        name: str,
        value: float,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"reward.kv.{name} must be a finite number in [0, 1]")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"reward.kv.{name} must be a finite number in [0, 1]")

    @staticmethod
    def _validate_positive_number(
        name: str,
        value: float,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"reward.kv.{name} must be a positive finite number")
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"reward.kv.{name} must be a positive finite number")

    @staticmethod
    def _validate_weight_sum(
        first_name: str,
        first_value: float,
        second_name: str,
        second_value: float,
    ) -> None:
        if not math.isclose(
            first_value + second_value,
            1.0,
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            raise ValueError(f"reward.kv.{first_name} + {second_name} must equal 1.0")

    @staticmethod
    def _normalize_allowed_sibling_keys(
        allowed_sibling_keys: List[str],
    ) -> Set[str]:
        if isinstance(allowed_sibling_keys, (str, bytes)):
            raise ValueError(
                "reward.kv.allowed_sibling_keys must contain stripped non-empty strings"
            )
        if not isinstance(allowed_sibling_keys, (list, tuple, set, ListConfig)):
            raise ValueError(
                "reward.kv.allowed_sibling_keys must contain stripped non-empty strings"
            )
        return set(allowed_sibling_keys)

    def _extract_json_text_and_format_cap(
        self,
        generation: str,
    ) -> Tuple[str, float]:
        extracted_answer = self.extract_answer_from_generation(generation=generation)
        extracted_answer = self.split_on_keywords(text=extracted_answer)
        stripped_answer = extracted_answer.strip()
        stop_split = self._find_parseable_stop_split(text=stripped_answer)

        if stop_split is None:
            if self._parse_prediction_json(text=stripped_answer) is not None:
                return stripped_answer, self.missing_stop_cap
            if self.required_stop_token in stripped_answer:
                return stripped_answer, self.middle_or_multiple_stop_cap
            return stripped_answer, self.missing_stop_cap

        json_text, trailing_text = stop_split
        if not trailing_text:
            return json_text, 1.0
        if self.required_stop_token in trailing_text:
            return json_text, self.middle_or_multiple_stop_cap
        return json_text, self.trailing_text_cap

    def _find_parseable_stop_split(
        self,
        text: str,
    ) -> Optional[Tuple[str, str]]:
        stop_indices = [
            match.start()
            for match in re.finditer(
                re.escape(self.required_stop_token),
                text,
            )
        ]
        for stop_index in reversed(stop_indices):
            json_text = text[:stop_index].rstrip()
            if self._parse_prediction_json(text=json_text) is None:
                continue
            trailing_text = text[stop_index + len(self.required_stop_token) :].strip()
            return json_text, trailing_text
        return None

    def _parse_prediction_json(
        self,
        text: str,
    ) -> Optional[Any]:
        if self.strict_json:
            return self._try_parse_strict_json(text=text)
        return self._try_parse_json(text=text)

    def _parse_solution_json(
        self,
        text: str,
    ) -> Optional[Any]:
        parsed = self._try_parse_strict_json(text=text)
        if parsed is not None:
            return parsed
        stop_split = self._find_parseable_stop_split(text=text.strip())
        if stop_split is None:
            return None
        json_text, _ = stop_split
        return self._try_parse_strict_json(text=json_text)

    @staticmethod
    def _try_parse_strict_json(text: str) -> Optional[Any]:
        stripped_text = text.strip()
        try:
            return json.loads(stripped_text)
        except json.JSONDecodeError:
            pass

        fenced = re.fullmatch(
            r"```(?:json)?\s*(\{.*\})\s*```",
            stripped_text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _try_parse_json(text: str) -> Optional[Any]:
        try:
            return json.loads(text)
        except Exception:
            pass
        if not isinstance(text, str):
            return None
        fenced = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except Exception:
                pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
        return None

    @staticmethod
    def _resolve_gt_root_name(
        gt_json: Any,
    ) -> Optional[str]:
        if not isinstance(gt_json, dict):
            return None
        if isinstance(gt_json.get("kv"), dict):
            return "kv"
        if isinstance(gt_json.get("tables"), dict):
            return "tables"
        if isinstance(gt_json.get("results"), list):
            return "results"
        return None

    def _compute_root_cap(
        self,
        pred_json: Any,
        root_name: str,
    ) -> float:
        if not isinstance(pred_json, dict):
            return self.root_mismatch_cap
        allowed_keys = {root_name} | self.allowed_sibling_keys
        if set(pred_json.keys()) - allowed_keys:
            return self.root_mismatch_cap
        if root_name == "results":
            if isinstance(pred_json.get(root_name), list):
                return 1.0
            return self.root_mismatch_cap
        if isinstance(pred_json.get(root_name), dict):
            return 1.0
        return self.root_mismatch_cap

    def _compute_length_cap(
        self,
        pred_json: Any,
        gt_json: Any,
        root_name: str,
    ) -> float:
        cap = 1.0
        pred_root = pred_json.get(root_name) if isinstance(pred_json, dict) else None
        gt_root = gt_json.get(root_name) if isinstance(gt_json, dict) else None
        pred_serialized = json.dumps(
            pred_root,
            ensure_ascii=False,
            sort_keys=True,
        )
        gt_serialized = json.dumps(
            gt_root,
            ensure_ascii=False,
            sort_keys=True,
        )
        if (
            len(pred_serialized) / max(len(gt_serialized), 1)
            > self.max_serialized_length_ratio
        ):
            cap = min(cap, self.length_ratio_cap)

        if root_name == "results":
            pred_leaf_count = self._count_results_leaf_values(results=pred_root)
            gt_leaf_count = self._count_results_leaf_values(results=gt_root)
        else:
            pred_leaf_count = self._count_leaf_values(node=pred_root)
            gt_leaf_count = self._count_leaf_values(node=gt_root)
        if (
            gt_leaf_count > 0
            and pred_leaf_count / gt_leaf_count > self.max_leaf_count_ratio
        ):
            cap = min(cap, self.leaf_count_ratio_cap)
        return cap

    def _compute_structure_cap(
        self,
        pred_json: Any,
        gt_json: Any,
        root_name: str,
    ) -> float:
        pred_root = pred_json.get(root_name) if isinstance(pred_json, dict) else None
        gt_root = gt_json.get(root_name) if isinstance(gt_json, dict) else None
        if root_name == "kv":
            return self._compute_kv_structure_cap(
                pred_kv=pred_root,
                gt_kv=gt_root,
            )
        if root_name == "results":
            return self._compute_results_structure_cap(
                pred_results=pred_root,
                gt_results=gt_root,
            )
        return self._compute_table_structure_cap(
            pred_tables=pred_root,
            gt_tables=gt_root,
        )

    def _compute_kv_structure_cap(
        self,
        pred_kv: Any,
        gt_kv: Any,
    ) -> float:
        pred_items = self._flatten_kv_items(node=pred_kv)
        gt_items = self._flatten_kv_items(node=gt_kv)
        cap = self._compute_leaf_balance_cap(
            pred_items=pred_items,
            gt_items=gt_items,
        )
        if self._has_duplicate_paths(items=pred_items):
            cap = min(
                cap,
                self.duplicate_path_cap,
            )
        if self._has_excess_duplicate_values(
            pred_items=pred_items,
            gt_items=gt_items,
        ):
            cap = min(
                cap,
                self.duplicate_value_cap,
            )
        if not self._has_top_level_path_overlap(
            pred_items=pred_items,
            gt_items=gt_items,
        ):
            cap = min(
                cap,
                self.kv_top_level_path_mismatch_cap,
            )
        return cap

    def _compute_table_structure_cap(
        self,
        pred_tables: Any,
        gt_tables: Any,
    ) -> float:
        pred_items = self._flatten_table_items(tables=pred_tables)
        gt_items = self._flatten_table_items(tables=gt_tables)
        cap = self._compute_leaf_balance_cap(
            pred_items=pred_items,
            gt_items=gt_items,
        )
        if not isinstance(pred_tables, dict) or not isinstance(gt_tables, dict):
            return cap
        if self._table_name_sets(tables=pred_tables) != self._table_name_sets(
            tables=gt_tables,
        ):
            cap = min(
                cap,
                self.table_name_mismatch_cap,
            )
        if self._has_table_column_mismatch(
            pred_tables=pred_tables,
            gt_tables=gt_tables,
        ):
            cap = min(
                cap,
                self.table_column_mismatch_cap,
            )
        if self._has_table_row_count_mismatch(
            pred_tables=pred_tables,
            gt_tables=gt_tables,
        ):
            cap = min(
                cap,
                self.table_row_count_mismatch_cap,
            )
        if self._has_duplicate_table_rows(
            pred_tables=pred_tables,
            gt_tables=gt_tables,
        ):
            cap = min(
                cap,
                self.duplicate_table_row_cap,
            )
        if self._has_excess_duplicate_values(
            pred_items=pred_items,
            gt_items=gt_items,
        ):
            cap = min(
                cap,
                self.duplicate_value_cap,
            )
        return cap

    def _compute_leaf_balance_cap(
        self,
        pred_items: List[Tuple[Tuple[str, ...], int, Any]],
        gt_items: List[Tuple[Tuple[str, ...], int, Any]],
    ) -> float:
        cap = 1.0
        pred_count = len(pred_items)
        gt_count = len(gt_items)
        if gt_count == 0:
            return cap
        extra_ratio = (
            max(
                pred_count - gt_count,
                0,
            )
            / gt_count
        )
        missing_ratio = (
            max(
                gt_count - pred_count,
                0,
            )
            / gt_count
        )
        if extra_ratio > self.extra_leaf_ratio_threshold:
            cap = min(
                cap,
                self.extra_leaf_cap,
            )
        if missing_ratio > self.missing_leaf_ratio_threshold:
            cap = min(
                cap,
                self.missing_leaf_cap,
            )
        return cap

    def _compute_root_score(
        self,
        pred_json: Any,
        gt_json: Any,
        root_name: str,
    ) -> float:
        pred_root = pred_json.get(root_name) if isinstance(pred_json, dict) else None
        gt_root = gt_json.get(root_name) if isinstance(gt_json, dict) else None
        if root_name == "kv":
            return self._compute_kv_root_score(
                pred_kv=pred_root,
                gt_kv=gt_root,
            )
        if root_name == "results":
            return self._compute_results_root_score(
                pred_results=pred_root,
                gt_results=gt_root,
            )
        return self._compute_table_root_score(
            pred_tables=pred_root,
            gt_tables=gt_root,
        )

    def _compute_kv_root_score(
        self,
        pred_kv: Any,
        gt_kv: Any,
    ) -> float:
        pred_items = self._flatten_kv_items(node=pred_kv)
        gt_items = self._flatten_kv_items(node=gt_kv)
        value_score = self._compute_fuzzy_f1(
            pred_items=pred_items,
            gt_items=gt_items,
            match_mode="value",
        )
        path_score = self._compute_fuzzy_f1(
            pred_items=pred_items,
            gt_items=gt_items,
            match_mode="path_value",
        )
        return self._clip_unit_score(
            score=self.kv_value_weight * value_score + self.kv_path_weight * path_score
        )

    def _compute_table_root_score(
        self,
        pred_tables: Any,
        gt_tables: Any,
    ) -> float:
        pred_items = self._flatten_table_items(tables=pred_tables)
        gt_items = self._flatten_table_items(tables=gt_tables)
        value_score = self._compute_fuzzy_f1(
            pred_items=pred_items,
            gt_items=gt_items,
            match_mode="row_value",
        )
        structure_score = self._compute_fuzzy_f1(
            pred_items=pred_items,
            gt_items=gt_items,
            match_mode="path_row_value",
        )
        return self._clip_unit_score(
            score=self.table_value_weight * value_score
            + self.table_structure_weight * structure_score
        )

    def _compute_results_structure_cap(
        self,
        pred_results: Any,
        gt_results: Any,
    ) -> float:
        pred_facts = self._normalize_results_items(results=pred_results)
        gt_facts = self._normalize_results_items(results=gt_results)
        cap = self._compute_leaf_balance_cap(
            pred_items=pred_facts["items"],
            gt_items=gt_facts["items"],
        )
        if pred_facts["invalid_count"] > 0 or gt_facts["invalid_count"] > 0:
            cap = min(
                cap,
                self.root_mismatch_cap,
            )
        if pred_facts["duplicate_count"] > 0 or gt_facts["duplicate_count"] > 0:
            cap = min(
                cap,
                self.duplicate_path_cap,
            )
        return cap

    def _compute_results_root_score(
        self,
        pred_results: Any,
        gt_results: Any,
    ) -> float:
        pred_items = self._normalize_results_items(results=pred_results)["items"]
        gt_items = self._normalize_results_items(results=gt_results)["items"]
        return self._compute_fuzzy_f1(
            pred_items=pred_items,
            gt_items=gt_items,
            match_mode="path_value",
        )

    def _count_results_leaf_values(
        self,
        results: Any,
    ) -> int:
        return len(self._normalize_results_items(results=results)["items"])

    def _normalize_results_items(
        self,
        results: Any,
    ) -> Dict[str, Any]:
        facts = {
            "items": [],
            "invalid_count": 0,
            "duplicate_count": 0,
        }
        if not isinstance(results, list):
            facts["invalid_count"] = 1
            return facts

        seen_target_ids: Set[str] = set()
        for item in results:
            normalized_item = self._normalize_result_item(item=item)
            if normalized_item is None:
                facts["invalid_count"] += 1
                continue
            target_id, text = normalized_item
            if target_id in seen_target_ids:
                facts["duplicate_count"] += 1
                continue
            seen_target_ids.add(target_id)
            facts["items"].append(
                (
                    (target_id,),
                    0,
                    text,
                )
            )
        return facts

    def _normalize_result_item(
        self,
        item: Any,
    ) -> Optional[Tuple[str, str]]:
        if not isinstance(item, dict):
            return None

        target_id = item.get("target_id")
        if isinstance(target_id, bool) or not isinstance(target_id, (str, int)):
            return None
        normalized_target_id = str(target_id).strip()
        if not normalized_target_id:
            return None

        text = item.get("text")
        if text is None:
            normalized_text = ""
        elif isinstance(text, bool):
            return None
        elif isinstance(text, (str, int, float)):
            normalized_text = str(text)
        else:
            return None
        return normalized_target_id, normalized_text

    def _flatten_kv_items(
        self,
        node: Any,
    ) -> List[Tuple[Tuple[str, ...], int, Any]]:
        leaves: List[Tuple[Tuple[str, ...], int, Any]] = []
        stack: List[Tuple[Any, List[str]]] = [
            (
                node,
                [],
            )
        ]
        while stack:
            obj, path = stack.pop()
            if isinstance(obj, dict):
                for key, value in reversed(list(obj.items())):
                    stack.append(
                        (
                            value,
                            path + [str(key)],
                        )
                    )
                continue
            if isinstance(obj, list):
                if any(isinstance(item, (dict, list)) for item in obj):
                    for item in reversed(obj):
                        stack.append(
                            (
                                item,
                                path,
                            )
                        )
                    continue
                for index, item in enumerate(obj):
                    leaves.append((tuple(path), index, item))
                if not obj:
                    leaves.append((tuple(path), 0, ""))
                continue
            leaves.append((tuple(path), 0, obj))
        return leaves

    def _flatten_table_items(
        self,
        tables: Any,
    ) -> List[Tuple[Tuple[str, ...], int, Any]]:
        if not isinstance(tables, dict):
            return []

        leaves: List[Tuple[Tuple[str, ...], int, Any]] = []
        for table_name in self._sorted_mapping_keys(mapping=tables):
            table_value = tables.get(table_name, {})
            rows = self._normalize_table_rows(
                rows=(
                    table_value.get("rows", []) if isinstance(table_value, dict) else []
                )
            )
            column_names = self._collect_table_column_names(rows=rows)
            for row_index, row in enumerate(rows):
                for column_name in column_names:
                    value = row.get(column_name, "") if isinstance(row, dict) else ""
                    leaves.append(
                        (
                            (
                                str(table_name),
                                str(column_name),
                            ),
                            row_index,
                            value,
                        )
                    )
            if not rows:
                leaves.append(((str(table_name),), 0, ""))
        return leaves

    @staticmethod
    def _normalize_table_rows(
        rows: Any,
    ) -> List[Any]:
        if isinstance(rows, list):
            return rows
        if not isinstance(rows, dict):
            return []
        return [rows[key] for key in KVReward._sorted_mapping_keys(mapping=rows)]

    @staticmethod
    def _sorted_mapping_keys(
        mapping: Dict[Any, Any],
    ) -> List[Any]:
        keys = list(mapping.keys())
        if all(KVReward._is_int_like(value=key) for key in keys):
            return sorted(
                keys,
                key=lambda key: int(str(key)),
            )
        return sorted(
            keys,
            key=lambda key: str(key),
        )

    @staticmethod
    def _is_int_like(
        value: Any,
    ) -> bool:
        try:
            int(str(value))
        except (TypeError, ValueError):
            return False
        return True

    def _collect_table_column_names(
        self,
        rows: List[Any],
    ) -> List[str]:
        column_names = []
        seen_columns = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in self._sorted_mapping_keys(mapping=row):
                column_name = str(key)
                if column_name in seen_columns:
                    continue
                column_names.append(column_name)
                seen_columns.add(column_name)
        return column_names

    def _has_duplicate_paths(
        self,
        items: List[Tuple[Tuple[str, ...], int, Any]],
    ) -> bool:
        path_counts: Dict[Tuple[Tuple[str, ...], int], int] = {}
        for path, index, _ in items:
            key = (path, index)
            path_counts[key] = (
                path_counts.get(
                    key,
                    0,
                )
                + 1
            )
            if path_counts[key] > 1:
                return True
        return False

    def _has_excess_duplicate_values(
        self,
        pred_items: List[Tuple[Tuple[str, ...], int, Any]],
        gt_items: List[Tuple[Tuple[str, ...], int, Any]],
    ) -> bool:
        pred_value_counts = self._count_normalized_values(items=pred_items)
        gt_value_counts = self._count_normalized_values(items=gt_items)
        for value, pred_count in pred_value_counts.items():
            allowed_count = max(
                gt_value_counts.get(
                    value,
                    0,
                ),
                1,
            )
            if pred_count > allowed_count:
                return True
        return False

    def _count_normalized_values(
        self,
        items: List[Tuple[Tuple[str, ...], int, Any]],
    ) -> Dict[str, int]:
        value_counts: Dict[str, int] = {}
        for _, _, value in items:
            normalized_value = self._coarse_normalize(
                text="" if value is None else str(value),
            )
            if normalized_value == "":
                continue
            value_counts[normalized_value] = (
                value_counts.get(
                    normalized_value,
                    0,
                )
                + 1
            )
        return value_counts

    def _has_top_level_path_overlap(
        self,
        pred_items: List[Tuple[Tuple[str, ...], int, Any]],
        gt_items: List[Tuple[Tuple[str, ...], int, Any]],
    ) -> bool:
        pred_roots = self._collect_top_level_path_names(items=pred_items)
        gt_roots = self._collect_top_level_path_names(items=gt_items)
        if not pred_roots or not gt_roots:
            return True
        return bool(pred_roots & gt_roots)

    @staticmethod
    def _collect_top_level_path_names(
        items: List[Tuple[Tuple[str, ...], int, Any]],
    ) -> Set[str]:
        return {path[0] for path, _, _ in items if path}

    def _table_name_sets(
        self,
        tables: Dict[Any, Any],
    ) -> Set[str]:
        return {str(key) for key in self._sorted_mapping_keys(mapping=tables)}

    def _has_table_column_mismatch(
        self,
        pred_tables: Dict[Any, Any],
        gt_tables: Dict[Any, Any],
    ) -> bool:
        for table_name in self._table_name_sets(
            tables=pred_tables
        ) & self._table_name_sets(
            tables=gt_tables,
        ):
            pred_columns = self._table_column_set(
                tables=pred_tables,
                table_name=table_name,
            )
            gt_columns = self._table_column_set(
                tables=gt_tables,
                table_name=table_name,
            )
            if pred_columns != gt_columns:
                return True
        return False

    def _table_column_set(
        self,
        tables: Dict[Any, Any],
        table_name: str,
    ) -> Set[str]:
        table_value = self._get_table_value_by_name(
            tables=tables,
            table_name=table_name,
        )
        rows = self._normalize_table_rows(
            rows=(
                table_value.get(
                    "rows",
                    [],
                )
                if isinstance(table_value, dict)
                else []
            ),
        )
        return set(self._collect_table_column_names(rows=rows))

    def _has_table_row_count_mismatch(
        self,
        pred_tables: Dict[Any, Any],
        gt_tables: Dict[Any, Any],
    ) -> bool:
        for table_name in self._table_name_sets(
            tables=pred_tables
        ) & self._table_name_sets(
            tables=gt_tables,
        ):
            if self._table_row_count(
                tables=pred_tables,
                table_name=table_name,
            ) != self._table_row_count(
                tables=gt_tables,
                table_name=table_name,
            ):
                return True
        return False

    def _table_row_count(
        self,
        tables: Dict[Any, Any],
        table_name: str,
    ) -> int:
        table_value = self._get_table_value_by_name(
            tables=tables,
            table_name=table_name,
        )
        rows = self._normalize_table_rows(
            rows=(
                table_value.get(
                    "rows",
                    [],
                )
                if isinstance(table_value, dict)
                else []
            ),
        )
        return len(rows)

    def _has_duplicate_table_rows(
        self,
        pred_tables: Dict[Any, Any],
        gt_tables: Dict[Any, Any],
    ) -> bool:
        for table_name in self._table_name_sets(
            tables=pred_tables
        ) & self._table_name_sets(
            tables=gt_tables,
        ):
            pred_row_counts = self._count_table_row_signatures(
                tables=pred_tables,
                table_name=table_name,
            )
            gt_row_counts = self._count_table_row_signatures(
                tables=gt_tables,
                table_name=table_name,
            )
            for signature, pred_count in pred_row_counts.items():
                allowed_count = max(
                    gt_row_counts.get(
                        signature,
                        0,
                    ),
                    1,
                )
                if pred_count > allowed_count:
                    return True
        return False

    def _count_table_row_signatures(
        self,
        tables: Dict[Any, Any],
        table_name: str,
    ) -> Dict[Tuple[Tuple[str, str], ...], int]:
        table_value = self._get_table_value_by_name(
            tables=tables,
            table_name=table_name,
        )
        rows = self._normalize_table_rows(
            rows=(
                table_value.get(
                    "rows",
                    [],
                )
                if isinstance(table_value, dict)
                else []
            ),
        )
        signature_counts: Dict[Tuple[Tuple[str, str], ...], int] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            signature = tuple(
                (
                    str(key),
                    self._coarse_normalize(
                        text=str(
                            row.get(
                                key,
                                "",
                            )
                        ),
                    ),
                )
                for key in self._sorted_mapping_keys(mapping=row)
            )
            signature_counts[signature] = (
                signature_counts.get(
                    signature,
                    0,
                )
                + 1
            )
        return signature_counts

    @staticmethod
    def _get_table_value_by_name(
        tables: Dict[Any, Any],
        table_name: str,
    ) -> Any:
        for key, value in tables.items():
            if str(key) == table_name:
                return value
        return {}

    def _compute_fuzzy_f1(
        self,
        pred_items: List[Tuple[Tuple[str, ...], int, Any]],
        gt_items: List[Tuple[Tuple[str, ...], int, Any]],
        match_mode: str,
    ) -> float:
        pred_count = len(pred_items)
        gt_count = len(gt_items)
        if pred_count == 0 and gt_count == 0:
            return 1.0
        if pred_count == 0 or gt_count == 0:
            return 0.0

        candidate_matches: List[Tuple[float, int, int]] = []
        for gt_index, gt_item in enumerate(gt_items):
            for pred_index, pred_item in enumerate(pred_items):
                score = self._compute_item_similarity(
                    pred_item=pred_item,
                    gt_item=gt_item,
                    match_mode=match_mode,
                )
                if self._score_passes_match_threshold(score=score):
                    candidate_matches.append(
                        (
                            score,
                            gt_index,
                            pred_index,
                        )
                    )

        candidate_matches.sort(
            key=lambda item: item[0],
            reverse=True,
        )
        matched_gt = set()
        matched_pred = set()
        matched_score_sum = 0.0
        match_count = 0
        for score, gt_index, pred_index in candidate_matches:
            if gt_index in matched_gt or pred_index in matched_pred:
                continue
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
            match_count += 1
            matched_score_sum += score

        if self.use_weighted_fuzzy_f1:
            precision = matched_score_sum / pred_count
            recall = matched_score_sum / gt_count
        else:
            precision = match_count / pred_count
            recall = match_count / gt_count
        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def _compute_item_similarity(
        self,
        pred_item: Tuple[Tuple[str, ...], int, Any],
        gt_item: Tuple[Tuple[str, ...], int, Any],
        match_mode: str,
    ) -> float:
        pred_path, pred_index, pred_value = pred_item
        gt_path, gt_index, gt_value = gt_item
        if match_mode == "path_value" and pred_path != gt_path:
            return 0.0
        if match_mode == "row_value" and pred_index != gt_index:
            return 0.0
        if match_mode == "path_row_value" and (
            pred_path != gt_path or pred_index != gt_index
        ):
            return 0.0
        return self._value_similarity(
            pred_value=pred_value,
            gt_value=gt_value,
        )

    def _clean_text(
        self,
        text: str,
    ) -> str:
        text = unicodedata.normalize(
            "NFKC",
            text,
        )
        text = text.strip()
        text = re.sub(
            r"[“”\"'`]",
            "",
            text,
        )
        text = re.sub(
            r"\s+",
            " ",
            text,
        )
        text = text.strip(".,;:!?()[]{}")
        return text.lower()

    def _coarse_normalize(
        self,
        text: str,
    ) -> str:
        text = self._clean_text(text=text)
        text = re.sub(
            r"[^\w가-힣%]+",
            "",
            text,
        )
        return text

    def _value_similarity(
        self,
        pred_value: Any,
        gt_value: Any,
    ) -> float:
        if (pred_value is None) != (gt_value is None):
            return self.empty_value_mismatch_cap
        pred_text = "" if pred_value is None else str(pred_value)
        gt_text = "" if gt_value is None else str(gt_value)
        pred_literal = self._literal_clean_text(text=pred_text)
        gt_literal = self._literal_clean_text(text=gt_text)
        pred_clean = self._clean_text(
            text=pred_text,
        )
        gt_clean = self._clean_text(text=gt_text)
        pred_coarse = self._coarse_normalize(
            text=pred_text,
        )
        gt_coarse = self._coarse_normalize(
            text=gt_text,
        )
        if pred_literal == "" and gt_literal == "":
            return 1.0
        if pred_literal == "" or gt_literal == "":
            return 0.0
        if pred_literal == gt_literal:
            return 1.0
        if pred_clean == gt_clean:
            return min(
                self.normalized_match_cap,
                self.non_exact_value_cap,
            )
        if pred_coarse == gt_coarse:
            return self._coarse_match_similarity(
                pred_text=pred_text,
                gt_text=gt_text,
            )
        fuzzy_similarity = SequenceMatcher(
            a=pred_clean,
            b=gt_clean,
        ).ratio()
        return min(
            fuzzy_similarity,
            self.non_exact_value_cap,
        )

    def _score_passes_match_threshold(
        self,
        score: float,
    ) -> bool:
        if self.use_weighted_fuzzy_f1:
            return score >= self.weighted_fuzzy_match_threshold
        return score >= self.score_threshold

    @staticmethod
    def _literal_clean_text(
        text: str,
    ) -> str:
        return text.strip()

    def _coarse_match_similarity(
        self,
        pred_text: str,
        gt_text: str,
    ) -> float:
        cap = self.coarse_match_cap
        cap = min(
            cap,
            self.non_exact_value_cap,
        )
        if self._has_numeric_punctuation_mismatch(
            pred_text=pred_text,
            gt_text=gt_text,
        ):
            cap = min(
                cap,
                self.numeric_punctuation_mismatch_cap,
            )
        return cap

    @staticmethod
    def _has_numeric_punctuation_mismatch(
        pred_text: str,
        gt_text: str,
    ) -> bool:
        if not re.search(r"\d", pred_text) or not re.search(r"\d", gt_text):
            return False
        pred_numeric_punctuation = set(
            re.findall(
                r"(?<=\d)[,.:](?=\d)",
                pred_text,
            )
        )
        gt_numeric_punctuation = set(
            re.findall(
                r"(?<=\d)[,.:](?=\d)",
                gt_text,
            )
        )
        return pred_numeric_punctuation != gt_numeric_punctuation

    def _count_leaf_values(
        self,
        node: Any,
    ) -> int:
        return len(self._flatten_kv_items(node=node))

    @staticmethod
    def _clip_unit_score(
        score: float,
    ) -> float:
        return min(max(float(score), 0.0), 1.0)
