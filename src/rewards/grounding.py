from typing import Dict, List, Tuple, Set, Optional, Any
import json
import re

from omegaconf import ListConfig

from .base import BaseReward


class GroundingBBoxReward(BaseReward):
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
        format_reward: float,
        schema_reward: float,
        page_reward: float,
        iou_weight: float,
        iou_05_threshold: float,
        iou_05_bonus: float,
        iou_07_threshold: float,
        iou_07_bonus: float,
        center_in_gt_bonus: float,
        large_box_area_threshold: float,
        large_box_penalty: float,
        hard_negative_iou_threshold: float,
        hard_negative_overlap_penalty: float,
        positive_duplicate_iou_threshold: float,
        min_reward: float,
        max_reward: float,
        schema_keys: Optional[Dict[str, List[str]]],
        status_values: Optional[Dict[str, List[str]]],
    ) -> None:
        super().__init__(
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
        self.format_reward = format_reward
        self.schema_reward = schema_reward
        self.page_reward = page_reward
        self.iou_weight = iou_weight
        self.iou_05_threshold = iou_05_threshold
        self.iou_05_bonus = iou_05_bonus
        self.iou_07_threshold = iou_07_threshold
        self.iou_07_bonus = iou_07_bonus
        self.center_in_gt_bonus = center_in_gt_bonus
        self.large_box_area_threshold = large_box_area_threshold
        self.large_box_penalty = large_box_penalty
        self.hard_negative_iou_threshold = hard_negative_iou_threshold
        self.hard_negative_overlap_penalty = hard_negative_overlap_penalty
        self.positive_duplicate_iou_threshold = positive_duplicate_iou_threshold
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.schema_keys = self._normalize_schema_keys(schema_keys=schema_keys)
        self.status_values = self._normalize_status_values(
            status_values=status_values,
        )

    def compute(
        self,
        completions: List[List[Dict[str, str]]],
        solution: List[Any],
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

            label = self._parse_label(solution=sol)
            if label is None:
                rewards.append(None)
                continue

            label_status = self._normalize_grounding_status(
                status=self._get_schema_value(
                    payload=label,
                    logical_key="grounding_status",
                )
            )
            if label_status != "found":
                rewards.append(
                    self._compute_negative_grounding_reward(
                        content=content,
                        label=label,
                    )
                )
                continue

            positive_boxes = self._collect_target_boxes(label=label)
            if not positive_boxes:
                rewards.append(None)
                continue

            extracted_answer = self.extract_answer_from_generation(generation=content)
            prediction = self._try_parse_json(text=extracted_answer)
            if prediction is None:
                rewards.append(0.0)
                continue

            reward = self.format_reward
            pred_boxes = self._collect_prediction_boxes(prediction=prediction)
            if self._is_schema_valid(
                prediction=prediction,
                pred_boxes=pred_boxes,
                label=label,
            ):
                reward += self.schema_reward

            if pred_boxes and self._has_page_match(
                pred_boxes=pred_boxes,
                positive_boxes=positive_boxes,
            ):
                reward += self.page_reward

            match = self._find_best_match(
                pred_boxes=pred_boxes,
                positive_boxes=positive_boxes,
            )
            if match is not None:
                iou, pred_box, target_box = match
                reward += self.iou_weight * iou
                if iou >= self.iou_05_threshold:
                    reward += self.iou_05_bonus
                if iou >= self.iou_07_threshold:
                    reward += self.iou_07_bonus
                if self._center_inside(
                    inner_box=pred_box["bbox"],
                    outer_box=target_box["bbox"],
                ):
                    reward += self.center_in_gt_bonus

            if self._has_large_box(pred_boxes=pred_boxes):
                reward += self.large_box_penalty

            if self._has_hard_negative_overlap(
                pred_boxes=pred_boxes,
                label=label,
                positive_boxes=positive_boxes,
            ):
                reward += self.hard_negative_overlap_penalty

            rewards.append(self._clip_reward(reward=reward))

        return rewards

    def _compute_negative_grounding_reward(
        self,
        content: str,
        label: Dict[str, Any],
    ) -> float:
        extracted_answer = self.extract_answer_from_generation(generation=content)
        prediction = self._try_parse_json(text=extracted_answer)
        if prediction is None:
            return 0.0

        reward = self.format_reward
        pred_boxes = self._collect_prediction_boxes(prediction=prediction)
        is_schema_valid = self._is_negative_schema_valid(
            prediction=prediction,
            pred_boxes=pred_boxes,
            label=label,
        )
        if is_schema_valid:
            reward += self.schema_reward

        if (
            is_schema_valid
            and self._normalize_grounding_status(
                status=self._get_schema_value(
                    payload=prediction,
                    logical_key="grounding_status",
                )
            )
            != "found"
            and not pred_boxes
        ):
            return self.max_reward

        if self._has_large_box(pred_boxes=pred_boxes):
            reward += self.large_box_penalty

        if self._has_hard_negative_overlap(
            pred_boxes=pred_boxes,
            label=label,
            positive_boxes=[],
        ):
            reward += self.hard_negative_overlap_penalty

        return self._clip_reward(reward=reward)

    @staticmethod
    def _try_parse_json(text: str) -> Optional[Any]:
        if not isinstance(text, str):
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fenced = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None

    def _parse_label(
        self,
        solution: Any,
    ) -> Optional[Dict[str, Any]]:
        if isinstance(solution, dict):
            return solution
        parsed = self._try_parse_json(text=solution)
        if isinstance(parsed, dict):
            return parsed
        return None

    def _get_schema_value(
        self,
        payload: Dict[str, Any],
        logical_key: str,
    ) -> Any:
        if not isinstance(payload, dict):
            return None
        aliases = self.schema_keys[logical_key]
        for alias in aliases:
            if alias in payload:
                return payload[alias]
        return None

    def _normalize_grounding_status(
        self,
        status: Any,
    ) -> Optional[str]:
        if not isinstance(status, str):
            return None
        normalized = status.strip().lower()
        for logical_status, aliases in self.status_values.items():
            if normalized in aliases:
                return logical_status
        return None

    def _normalize_schema_keys(
        self,
        schema_keys: Optional[Dict[str, List[str]]],
    ) -> Dict[str, List[str]]:
        config = schema_keys or self._default_schema_keys()
        required_keys = self._default_schema_keys().keys()
        normalized: Dict[str, List[str]] = {}
        for logical_key in required_keys:
            if logical_key not in config:
                raise ValueError(
                    f"grounding_bbox.schema_keys missing key: {logical_key}"
                )
            aliases = config[logical_key]
            normalized[logical_key] = self._normalize_aliases(
                aliases=aliases,
                config_name=f"grounding_bbox.schema_keys.{logical_key}",
            )
        return normalized

    def _normalize_status_values(
        self,
        status_values: Optional[Dict[str, List[str]]],
    ) -> Dict[str, List[str]]:
        config = status_values or self._default_status_values()
        normalized: Dict[str, List[str]] = {}
        for logical_status in ["found", "not_found"]:
            if logical_status not in config:
                raise ValueError(
                    f"grounding_bbox.status_values missing key: {logical_status}"
                )
            normalized[logical_status] = [
                alias.lower()
                for alias in self._normalize_aliases(
                    aliases=config[logical_status],
                    config_name=f"grounding_bbox.status_values.{logical_status}",
                )
            ]
        return normalized

    @staticmethod
    def _normalize_aliases(
        aliases: Any,
        config_name: str,
    ) -> List[str]:
        if not isinstance(aliases, (list, ListConfig)):
            raise ValueError(f"{config_name} must be a non-empty list of strings.")
        normalized = []
        for alias in aliases:
            if not isinstance(alias, str) or alias.strip() == "":
                raise ValueError(f"{config_name} must contain only non-empty strings.")
            normalized.append(alias.strip())
        if not normalized:
            raise ValueError(f"{config_name} must be a non-empty list of strings.")
        return normalized

    @staticmethod
    def _default_schema_keys() -> Dict[str, List[str]]:
        return {
            "field_path": ["field_path"],
            "value_index": ["value_index"],
            "grounding_status": ["grounding_status"],
            "prediction_occurrences": [
                "evidence_occurrences",
                "positive_occurrences",
                "occurrences",
            ],
            "label_occurrences": [
                "positive_occurrences",
                "evidence_occurrences",
                "occurrences",
            ],
            "hard_negative_evidence": ["hard_negative_evidence"],
            "fragments": ["fragments"],
            "page": ["page"],
            "bbox": ["bbox"],
            "envelope_bbox": [
                "envelope_bbox",
                "bbox",
            ],
            "coord_system": ["coord_system"],
        }

    @staticmethod
    def _default_status_values() -> Dict[str, List[str]]:
        return {
            "found": ["found"],
            "not_found": [
                "not_found",
                "missing",
                "absent",
            ],
        }

    def _is_schema_valid(
        self,
        prediction: Any,
        pred_boxes: List[Dict[str, Any]],
        label: Dict[str, Any],
    ) -> bool:
        if not isinstance(prediction, dict):
            return False
        if not isinstance(
            self._get_schema_value(
                payload=prediction,
                logical_key="field_path",
            ),
            str,
        ):
            return False
        if (
            self._normalize_grounding_status(
                status=self._get_schema_value(
                    payload=prediction,
                    logical_key="grounding_status",
                )
            )
            is None
        ):
            return False
        if not isinstance(
            self._get_schema_value(
                payload=prediction,
                logical_key="prediction_occurrences",
            ),
            list,
        ):
            return False
        if not pred_boxes:
            return False

        coord_system = self._get_schema_value(
            payload=label,
            logical_key="coord_system",
        )
        if not isinstance(coord_system, str):
            return True

        return all(box.get("coord_system") == coord_system for box in pred_boxes)

    def _is_negative_schema_valid(
        self,
        prediction: Any,
        pred_boxes: List[Dict[str, Any]],
        label: Dict[str, Any],
    ) -> bool:
        if not isinstance(prediction, dict):
            return False
        if not isinstance(
            self._get_schema_value(
                payload=prediction,
                logical_key="field_path",
            ),
            str,
        ):
            return False
        if (
            self._normalize_grounding_status(
                status=self._get_schema_value(
                    payload=prediction,
                    logical_key="grounding_status",
                )
            )
            is None
        ):
            return False
        if not isinstance(
            self._get_schema_value(
                payload=prediction,
                logical_key="prediction_occurrences",
            ),
            list,
        ):
            return False

        coord_system = self._get_schema_value(
            payload=label,
            logical_key="coord_system",
        )
        if not isinstance(coord_system, str):
            return True
        return all(box.get("coord_system") == coord_system for box in pred_boxes)

    def _collect_prediction_boxes(
        self,
        prediction: Any,
    ) -> List[Dict[str, Any]]:
        if not isinstance(prediction, dict):
            return []

        occurrences = self._get_schema_value(
            payload=prediction,
            logical_key="prediction_occurrences",
        )
        if not isinstance(occurrences, list):
            return []

        boxes: List[Dict[str, Any]] = []
        for occurrence in occurrences:
            boxes.extend(self._collect_occurrence_boxes(occurrence=occurrence))
        return boxes

    def _collect_target_boxes(
        self,
        label: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        occurrences = self._get_schema_value(
            payload=label,
            logical_key="label_occurrences",
        )
        if not isinstance(occurrences, list):
            return []

        boxes: List[Dict[str, Any]] = []
        for occurrence in occurrences:
            boxes.extend(self._collect_occurrence_boxes(occurrence=occurrence))
        return boxes

    def _collect_occurrence_boxes(
        self,
        occurrence: Any,
    ) -> List[Dict[str, Any]]:
        if not isinstance(occurrence, dict):
            return []

        page = self._get_schema_value(
            payload=occurrence,
            logical_key="page",
        )
        fragments = self._get_schema_value(
            payload=occurrence,
            logical_key="fragments",
        )
        if isinstance(fragments, list):
            fragment_boxes = []
            for fragment in fragments:
                if not isinstance(fragment, dict):
                    continue
                box = self._build_box_record(
                    page=page,
                    bbox=self._get_schema_value(
                        payload=fragment,
                        logical_key="bbox",
                    ),
                    coord_system=self._get_schema_value(
                        payload=fragment,
                        logical_key="coord_system",
                    ),
                )
                if box is not None:
                    fragment_boxes.append(box)
            if fragment_boxes:
                return fragment_boxes

        box = self._build_box_record(
            page=page,
            bbox=self._get_schema_value(
                payload=occurrence,
                logical_key="envelope_bbox",
            ),
            coord_system=self._get_schema_value(
                payload=occurrence,
                logical_key="coord_system",
            ),
        )
        if box is None:
            return []
        return [box]

    def _build_box_record(
        self,
        page: Any,
        bbox: Any,
        coord_system: Any,
    ) -> Optional[Dict[str, Any]]:
        parsed_bbox = self._parse_bbox(bbox=bbox)
        if parsed_bbox is None:
            return None
        if not isinstance(page, int):
            return None
        if not isinstance(coord_system, str):
            coord_system = None
        return {
            "page": page,
            "bbox": parsed_bbox,
            "coord_system": coord_system,
        }

    @staticmethod
    def _parse_bbox(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
        if not isinstance(bbox, list):
            return None
        if len(bbox) != 4:
            return None

        values = []
        for value in bbox:
            if not isinstance(value, (int, float)):
                return None
            values.append(float(value))

        x1, y1, x2, y2 = values
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _find_best_match(
        self,
        pred_boxes: List[Dict[str, Any]],
        positive_boxes: List[Dict[str, Any]],
    ) -> Optional[Tuple[float, Dict[str, Any], Dict[str, Any]]]:
        best_match: Optional[Tuple[float, Dict[str, Any], Dict[str, Any]]] = None

        for pred_box in pred_boxes:
            for positive_box in positive_boxes:
                if pred_box["page"] != positive_box["page"]:
                    continue
                iou = self._bbox_iou(
                    left=pred_box["bbox"],
                    right=positive_box["bbox"],
                )
                if best_match is None or iou > best_match[0]:
                    best_match = (
                        iou,
                        pred_box,
                        positive_box,
                    )

        return best_match

    @staticmethod
    def _has_page_match(
        pred_boxes: List[Dict[str, Any]],
        positive_boxes: List[Dict[str, Any]],
    ) -> bool:
        positive_pages = {box["page"] for box in positive_boxes}
        return any(box["page"] in positive_pages for box in pred_boxes)

    def _has_large_box(
        self,
        pred_boxes: List[Dict[str, Any]],
    ) -> bool:
        return any(
            self._bbox_area(bbox=box["bbox"]) > self.large_box_area_threshold
            for box in pred_boxes
        )

    def _has_hard_negative_overlap(
        self,
        pred_boxes: List[Dict[str, Any]],
        label: Dict[str, Any],
        positive_boxes: List[Dict[str, Any]],
    ) -> bool:
        hard_negative_boxes = self._collect_hard_negative_boxes(label=label)
        hard_negative_boxes = [
            box
            for box in hard_negative_boxes
            if not self._is_positive_duplicate(
                hard_negative_box=box,
                positive_boxes=positive_boxes,
            )
        ]

        for pred_box in pred_boxes:
            for hard_negative_box in hard_negative_boxes:
                if pred_box["page"] != hard_negative_box["page"]:
                    continue
                iou = self._bbox_iou(
                    left=pred_box["bbox"],
                    right=hard_negative_box["bbox"],
                )
                if iou >= self.hard_negative_iou_threshold:
                    return True
        return False

    def _collect_hard_negative_boxes(
        self,
        label: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        hard_negatives = self._get_schema_value(
            payload=label,
            logical_key="hard_negative_evidence",
        )
        if not isinstance(hard_negatives, list):
            return []

        boxes: List[Dict[str, Any]] = []
        for hard_negative in hard_negatives:
            if not isinstance(hard_negative, dict):
                continue
            box = self._build_box_record(
                page=self._get_schema_value(
                    payload=hard_negative,
                    logical_key="page",
                ),
                bbox=self._get_schema_value(
                    payload=hard_negative,
                    logical_key="bbox",
                ),
                coord_system=self._get_schema_value(
                    payload=label,
                    logical_key="coord_system",
                ),
            )
            if box is not None:
                boxes.append(box)
        return boxes

    def _is_positive_duplicate(
        self,
        hard_negative_box: Dict[str, Any],
        positive_boxes: List[Dict[str, Any]],
    ) -> bool:
        for positive_box in positive_boxes:
            if hard_negative_box["page"] != positive_box["page"]:
                continue
            iou = self._bbox_iou(
                left=hard_negative_box["bbox"],
                right=positive_box["bbox"],
            )
            if iou >= self.positive_duplicate_iou_threshold:
                return True
        return False

    @staticmethod
    def _bbox_iou(
        left: Tuple[float, float, float, float],
        right: Tuple[float, float, float, float],
    ) -> float:
        x1 = max(left[0], right[0])
        y1 = max(left[1], right[1])
        x2 = min(left[2], right[2])
        y2 = min(left[3], right[3])

        inter_width = max(0.0, x2 - x1)
        inter_height = max(0.0, y2 - y1)
        intersection = inter_width * inter_height / 1_000_000.0
        if intersection <= 0:
            return 0.0

        left_area = GroundingBBoxReward._bbox_area(bbox=left)
        right_area = GroundingBBoxReward._bbox_area(bbox=right)
        union = left_area + right_area - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    @staticmethod
    def _bbox_area(
        bbox: Tuple[float, float, float, float],
    ) -> float:
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / 1_000_000.0

    @staticmethod
    def _center_inside(
        inner_box: Tuple[float, float, float, float],
        outer_box: Tuple[float, float, float, float],
    ) -> bool:
        center_x = (inner_box[0] + inner_box[2]) / 2.0
        center_y = (inner_box[1] + inner_box[3]) / 2.0
        return (
            outer_box[0] <= center_x <= outer_box[2]
            and outer_box[1] <= center_y <= outer_box[3]
        )

    def _clip_reward(
        self,
        reward: float,
    ) -> float:
        return min(
            self.max_reward,
            max(
                self.min_reward,
                reward,
            ),
        )


class GroundingSelectionReward(BaseReward):
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
        format_reward: float,
        schema_reward: float,
        exact_match_reward: float,
        partial_match_weight: float,
        over_selection_penalty: float,
        wrong_selection_penalty: float,
        partial_match_cap: float,
        multi_fragment_partial_match_cap: float,
        missing_gold_id_penalty: float,
        single_id_partial_cap: float,
        short_multi_id_partial_cap: float,
        long_multi_id_partial_cap: float,
        very_long_multi_id_partial_cap: float,
        over_selection_cap: float,
        wrong_occurrence_cap: float,
        wrong_occurrence_overlap_threshold: float,
        extra_selected_id_penalty: float,
        schema_only_reward_cap: float,
        empty_selection_reward_cap: float,
        invalid_schema_reward_cap: float,
        extra_target_reward_cap: float,
        target_quality_aggregation: str,
        hard_target_min_gold_ids: int,
        hard_target_weight: float,
        min_reward: float,
        max_reward: float,
        schema_keys: Optional[Dict[str, List[str]]],
    ) -> None:
        super().__init__(
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
        self.format_reward = format_reward
        self.schema_reward = schema_reward
        self.exact_match_reward = exact_match_reward
        self.partial_match_weight = partial_match_weight
        self.over_selection_penalty = over_selection_penalty
        self.wrong_selection_penalty = wrong_selection_penalty
        self.partial_match_cap = partial_match_cap
        self.multi_fragment_partial_match_cap = multi_fragment_partial_match_cap
        self.missing_gold_id_penalty = missing_gold_id_penalty
        self.single_id_partial_cap = single_id_partial_cap
        self.short_multi_id_partial_cap = short_multi_id_partial_cap
        self.long_multi_id_partial_cap = long_multi_id_partial_cap
        self.very_long_multi_id_partial_cap = very_long_multi_id_partial_cap
        self.over_selection_cap = over_selection_cap
        self.wrong_occurrence_cap = wrong_occurrence_cap
        self.wrong_occurrence_overlap_threshold = wrong_occurrence_overlap_threshold
        self.extra_selected_id_penalty = extra_selected_id_penalty
        self.schema_only_reward_cap = schema_only_reward_cap
        self.empty_selection_reward_cap = empty_selection_reward_cap
        self.invalid_schema_reward_cap = invalid_schema_reward_cap
        self.extra_target_reward_cap = extra_target_reward_cap
        self.target_quality_aggregation = target_quality_aggregation
        self.hard_target_min_gold_ids = hard_target_min_gold_ids
        self.hard_target_weight = hard_target_weight
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.schema_keys = self._normalize_schema_keys(schema_keys=schema_keys)
        self._validate_target_quality_aggregation()

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

            label = self._parse_label(solution=sol)
            if label is None:
                rewards.append(None)
                continue

            gold_items = self._get_grounding_items(payload=label)
            if gold_items is None:
                rewards.append(None)
                continue

            gold_item_map, gold_invalid_count = self._build_item_map(
                items=gold_items,
            )
            if gold_invalid_count > 0:
                rewards.append(None)
                continue
            if not self._has_valid_gold_selections(item_map=gold_item_map):
                rewards.append(None)
                continue

            extracted_answer = self.extract_answer_from_generation(generation=content)
            prediction = GroundingBBoxReward._try_parse_json(text=extracted_answer)
            if prediction is None:
                rewards.append(0.0)
                continue

            rewards.append(
                self._compute_selection_reward(
                    prediction=prediction,
                    gold_item_map=gold_item_map,
                )
            )

        return rewards

    def _compute_selection_reward(
        self,
        prediction: Any,
        gold_item_map: Dict[str, Dict[str, Any]],
    ) -> float:
        if not isinstance(prediction, dict):
            return 0.0

        prediction_items = self._get_grounding_items(payload=prediction)
        if prediction_items is None:
            return self._clip_reward(reward=self.format_reward)

        prediction_item_map, prediction_invalid_count = self._build_item_map(
            items=prediction_items,
        )
        prediction_schema_valid = self._is_prediction_schema_valid(
            item_map=prediction_item_map,
            invalid_count=prediction_invalid_count,
        )

        base_reward = self.format_reward
        if prediction_schema_valid:
            base_reward += self.schema_reward

        quality = self._compute_target_quality(
            prediction_item_map=prediction_item_map,
            gold_item_map=gold_item_map,
        )
        reward = base_reward + max(0.0, self.max_reward - base_reward) * quality
        reward = self._apply_schema_only_policy(
            reward=reward,
            quality=quality,
            prediction_item_map=prediction_item_map,
        )
        reward = self._apply_extra_target_policy(
            reward=reward,
            prediction_item_map=prediction_item_map,
            gold_item_map=gold_item_map,
        )
        reward += self.wrong_selection_penalty * prediction_invalid_count
        reward = self._apply_invalid_schema_policy(
            reward=reward,
            prediction_schema_valid=prediction_schema_valid,
        )
        return self._clip_reward(reward=reward)

    def _compute_target_quality(
        self,
        prediction_item_map: Dict[str, Dict[str, Any]],
        gold_item_map: Dict[str, Dict[str, Any]],
    ) -> float:
        if not gold_item_map:
            return 1.0 if not prediction_item_map else 0.0

        target_qualities: List[Tuple[float, Dict[str, Any]]] = []
        for target_id, gold_item in gold_item_map.items():
            prediction_item = prediction_item_map.get(target_id)
            if prediction_item is None:
                target_qualities.append((0.0, gold_item))
                continue
            target_qualities.append(
                (
                    self._compute_item_quality(
                        prediction_item=prediction_item,
                        gold_item=gold_item,
                    ),
                    gold_item,
                )
            )

        return self._aggregate_target_qualities(target_qualities=target_qualities)

    def _compute_item_quality(
        self,
        prediction_item: Dict[str, Any],
        gold_item: Dict[str, Any],
    ) -> float:
        prediction_selection = self._normalize_selected_ids(item=prediction_item)
        gold_selection = self._normalize_selected_ids(item=gold_item)
        if prediction_selection is None or gold_selection is None:
            return 0.0

        prediction_ids, has_duplicate_ids = prediction_selection
        gold_ids, _ = gold_selection
        if has_duplicate_ids:
            return 0.0

        max_item_score = self.exact_match_reward + self.partial_match_weight
        if max_item_score <= 0:
            return 1.0 if prediction_ids == gold_ids else 0.0

        if prediction_ids == gold_ids:
            return 1.0

        quality = self._compute_non_exact_selection_quality(
            prediction_ids=prediction_ids,
            gold_ids=gold_ids,
            max_item_score=max_item_score,
        )
        missing_gold_ids = gold_ids - prediction_ids
        return self._apply_partial_selection_policy(
            quality=quality,
            prediction_ids=prediction_ids,
            gold_ids=gold_ids,
            missing_gold_count=len(missing_gold_ids),
            extra_selected_count=len(prediction_ids - gold_ids),
        )

    def _compute_non_exact_selection_quality(
        self,
        prediction_ids: Set[int],
        gold_ids: Set[int],
        max_item_score: float,
    ) -> float:
        extra_selected_count = len(prediction_ids - gold_ids)
        item_score = self.partial_match_weight * self._selection_f1(
            pred_selected_ids=prediction_ids,
            gold_selected_ids=gold_ids,
        )
        if extra_selected_count > 0:
            item_score += self.wrong_selection_penalty
        if len(prediction_ids) > len(gold_ids):
            item_score += self.over_selection_penalty

        quality = min(
            1.0,
            max(
                0.0,
                item_score / max_item_score,
            ),
        )
        return quality

    def _apply_partial_selection_policy(
        self,
        quality: float,
        prediction_ids: Set[int],
        gold_ids: Set[int],
        missing_gold_count: int,
        extra_selected_count: int,
    ) -> float:
        capped_quality = min(
            quality,
            self._get_partial_match_cap(gold_id_count=len(gold_ids)),
        )
        if missing_gold_count == 0:
            return self._apply_extra_selection_policy(
                quality=capped_quality,
                extra_selected_count=extra_selected_count,
            )

        if self._is_wrong_occurrence(
            prediction_ids=prediction_ids,
            gold_ids=gold_ids,
        ):
            capped_quality = min(
                capped_quality,
                self.wrong_occurrence_cap,
            )
        capped_quality += self.missing_gold_id_penalty * missing_gold_count
        capped_quality = self._apply_extra_selection_policy(
            quality=capped_quality,
            extra_selected_count=extra_selected_count,
        )
        return self._clip_unit_score(score=capped_quality)

    def _get_partial_match_cap(
        self,
        gold_id_count: int,
    ) -> float:
        cap = self.partial_match_cap
        if gold_id_count == 1:
            cap = min(
                cap,
                self.single_id_partial_cap,
            )
        elif gold_id_count <= 4:
            cap = min(
                cap,
                self.short_multi_id_partial_cap,
            )
        elif gold_id_count <= 8:
            cap = min(
                cap,
                self.long_multi_id_partial_cap,
            )
        else:
            cap = min(
                cap,
                self.very_long_multi_id_partial_cap,
            )
        if gold_id_count > 1:
            cap = min(
                cap,
                self.multi_fragment_partial_match_cap,
            )
        return cap

    def _apply_extra_selection_policy(
        self,
        quality: float,
        extra_selected_count: int,
    ) -> float:
        if extra_selected_count == 0:
            return self._clip_unit_score(score=quality)
        quality = min(
            quality,
            self.over_selection_cap,
        )
        quality += self.extra_selected_id_penalty * extra_selected_count
        return self._clip_unit_score(score=quality)

    def _apply_schema_only_policy(
        self,
        reward: float,
        quality: float,
        prediction_item_map: Dict[str, Dict[str, Any]],
    ) -> float:
        if quality > 0:
            return reward
        reward = min(
            reward,
            self.schema_only_reward_cap,
        )
        if not self._has_any_predicted_selection(item_map=prediction_item_map):
            reward = min(
                reward,
                self.empty_selection_reward_cap,
            )
        return reward

    def _apply_invalid_schema_policy(
        self,
        reward: float,
        prediction_schema_valid: bool,
    ) -> float:
        if prediction_schema_valid:
            return reward
        return min(
            reward,
            self.invalid_schema_reward_cap,
        )

    def _apply_extra_target_policy(
        self,
        reward: float,
        prediction_item_map: Dict[str, Dict[str, Any]],
        gold_item_map: Dict[str, Dict[str, Any]],
    ) -> float:
        extra_target_count = self._count_extra_targets(
            prediction_item_map=prediction_item_map,
            gold_item_map=gold_item_map,
        )
        if extra_target_count == 0:
            return reward
        reward += self.wrong_selection_penalty * extra_target_count
        return min(
            reward,
            self.extra_target_reward_cap,
        )

    def _has_any_predicted_selection(
        self,
        item_map: Dict[str, Dict[str, Any]],
    ) -> bool:
        for item in item_map.values():
            selected_ids = self._normalize_selected_ids(item=item)
            if selected_ids is None:
                continue
            ids, _ = selected_ids
            if ids:
                return True
        return False

    def _is_wrong_occurrence(
        self,
        prediction_ids: Set[int],
        gold_ids: Set[int],
    ) -> bool:
        if not prediction_ids or not gold_ids:
            return False
        overlap_ratio = len(prediction_ids & gold_ids) / len(gold_ids)
        return overlap_ratio <= self.wrong_occurrence_overlap_threshold

    def _aggregate_target_qualities(
        self,
        target_qualities: List[Tuple[float, Dict[str, Any]]],
    ) -> float:
        if self.target_quality_aggregation == "mean":
            return self._mean_target_quality(target_qualities=target_qualities)
        if self.target_quality_aggregation == "min":
            return min(quality for quality, _ in target_qualities)
        if self.target_quality_aggregation == "hard_weighted_mean":
            return self._hard_weighted_target_quality(
                target_qualities=target_qualities,
            )
        raise ValueError(
            f"Unsupported target_quality_aggregation: {self.target_quality_aggregation}"
        )

    def _mean_target_quality(
        self,
        target_qualities: List[Tuple[float, Dict[str, Any]]],
    ) -> float:
        return sum(quality for quality, _ in target_qualities) / len(target_qualities)

    def _hard_weighted_target_quality(
        self,
        target_qualities: List[Tuple[float, Dict[str, Any]]],
    ) -> float:
        weighted_sum = 0.0
        total_weight = 0.0
        for quality, gold_item in target_qualities:
            target_weight = self._get_target_quality_weight(gold_item=gold_item)
            weighted_sum += quality * target_weight
            total_weight += target_weight
        if total_weight <= 0:
            return 0.0
        return weighted_sum / total_weight

    def _get_target_quality_weight(
        self,
        gold_item: Dict[str, Any],
    ) -> float:
        selected_ids = self._normalize_selected_ids(item=gold_item)
        if selected_ids is None:
            return 1.0
        gold_ids, _ = selected_ids
        if len(gold_ids) >= self.hard_target_min_gold_ids:
            return self.hard_target_weight
        return 1.0

    def _validate_target_quality_aggregation(
        self,
    ) -> None:
        valid_modes = {
            "mean",
            "min",
            "hard_weighted_mean",
        }
        if self.target_quality_aggregation not in valid_modes:
            raise ValueError(
                "reward.grounding_selection.target_quality_aggregation must be one of "
                f"{sorted(valid_modes)}"
            )

    @staticmethod
    def _clip_unit_score(
        score: float,
    ) -> float:
        return min(
            1.0,
            max(
                0.0,
                score,
            ),
        )

    def _count_extra_targets(
        self,
        prediction_item_map: Dict[str, Dict[str, Any]],
        gold_item_map: Dict[str, Dict[str, Any]],
    ) -> int:
        return len(set(prediction_item_map.keys()) - set(gold_item_map.keys()))

    def _parse_label(
        self,
        solution: Any,
    ) -> Optional[Dict[str, Any]]:
        if isinstance(solution, dict):
            return solution
        parsed = GroundingBBoxReward._try_parse_json(text=solution)
        if isinstance(parsed, dict):
            return parsed
        return None

    def _get_grounding_items(
        self,
        payload: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        items = self._get_schema_value(
            payload=payload,
            logical_key="items",
        )
        if not isinstance(items, list):
            return None
        if not all(isinstance(item, dict) for item in items):
            return None
        return items

    def _build_item_map(
        self,
        items: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], int]:
        item_map: Dict[str, Dict[str, Any]] = {}
        invalid_count = 0
        for item in items:
            target_id = self._normalize_target_id(item=item)
            if target_id is None:
                invalid_count += 1
                continue
            if target_id in item_map:
                invalid_count += 1
                continue
            item_map[target_id] = item
        return item_map, invalid_count

    def _has_valid_gold_selections(
        self,
        item_map: Dict[str, Dict[str, Any]],
    ) -> bool:
        for item in item_map.values():
            selected_ids = self._normalize_selected_ids(item=item)
            if selected_ids is None:
                return False
            ids, has_duplicate_ids = selected_ids
            if has_duplicate_ids or not ids:
                return False
        return True

    def _is_prediction_schema_valid(
        self,
        item_map: Dict[str, Dict[str, Any]],
        invalid_count: int,
    ) -> bool:
        if invalid_count > 0:
            return False
        return all(
            self._is_prediction_item_schema_valid(item=item)
            for item in item_map.values()
        )

    def _is_prediction_item_schema_valid(
        self,
        item: Dict[str, Any],
    ) -> bool:
        selected_ids = self._normalize_selected_ids(item=item)
        if selected_ids is None:
            return False
        _, has_duplicate_ids = selected_ids
        return not has_duplicate_ids

    def _normalize_target_id(
        self,
        item: Dict[str, Any],
    ) -> Optional[str]:
        value = self._get_schema_value(
            payload=item,
            logical_key="target_id",
        )
        if isinstance(value, bool):
            return None
        if not isinstance(value, (str, int)):
            return None
        normalized = str(value).strip()
        return normalized if normalized else None

    def _get_schema_value(
        self,
        payload: Dict[str, Any],
        logical_key: str,
    ) -> Any:
        if not isinstance(payload, dict):
            return None
        aliases = self.schema_keys[logical_key]
        for alias in aliases:
            if alias in payload:
                return payload[alias]
        return None

    def _normalize_selected_ids(
        self,
        item: Dict[str, Any],
    ) -> Optional[Tuple[Set[int], bool]]:
        value = self._get_schema_value(
            payload=item,
            logical_key="selected_ids",
        )
        if not isinstance(value, list):
            value = [value]
        normalized_ids: List[int] = []
        for selected_id in value:
            normalized_id = self._normalize_selected_id(selected_id=selected_id)
            if normalized_id is not None:
                normalized_ids.append(normalized_id)
                continue
            return None
        selected_id_set = set(normalized_ids)
        return selected_id_set, len(selected_id_set) != len(normalized_ids)

    @staticmethod
    def _normalize_selected_id(
        selected_id: Any,
    ) -> Optional[int]:
        if isinstance(selected_id, bool):
            return None
        if isinstance(selected_id, int):
            return selected_id
        if isinstance(selected_id, str) and selected_id.strip().isdigit():
            return int(selected_id.strip())
        return None

    def _normalize_schema_keys(
        self,
        schema_keys: Optional[Dict[str, List[str]]],
    ) -> Dict[str, List[str]]:
        config = schema_keys or self._default_schema_keys()
        required_keys = self._default_schema_keys().keys()
        normalized: Dict[str, List[str]] = {}
        for logical_key in required_keys:
            if logical_key not in config:
                raise ValueError(
                    f"grounding_selection.schema_keys missing key: {logical_key}"
                )
            aliases = config[logical_key]
            normalized[logical_key] = GroundingBBoxReward._normalize_aliases(
                aliases=aliases,
                config_name=f"grounding_selection.schema_keys.{logical_key}",
            )
        return normalized

    @staticmethod
    def _selection_f1(
        pred_selected_ids: Set[int],
        gold_selected_ids: Set[int],
    ) -> float:
        if not pred_selected_ids or not gold_selected_ids:
            return 0.0
        true_positive = len(pred_selected_ids & gold_selected_ids)
        precision = true_positive / len(pred_selected_ids)
        recall = true_positive / len(gold_selected_ids)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _default_schema_keys() -> Dict[str, List[str]]:
        return {
            "items": ["grounding"],
            "target_id": ["target_id"],
            "selected_ids": [
                "selected_ids",
                "selected_candidate_ids",
            ],
        }

    def _clip_reward(
        self,
        reward: float,
    ) -> float:
        return min(
            self.max_reward,
            max(
                self.min_reward,
                reward,
            ),
        )
