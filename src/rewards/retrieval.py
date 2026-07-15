from typing import Dict, List, Tuple, Set, Union, Optional, Any
from ast import literal_eval
import math

from omegaconf import DictConfig, ListConfig

import numpy as np

from .base import BaseReward
from .embedding import VllmEmbedding
from .vector_store import FaissIndex


class RetrievalBaseReward(BaseReward):
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
        database: FaissIndex,
        embedding: VllmEmbedding,
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
        self.database = database
        self.embedding = embedding
        self._database_loaded = False

    def _ensure_ready(self) -> None:
        if not self._database_loaded:
            self.database.load()
            self._database_loaded = True

    def _get_original_and_rewritten_candidates(
        self,
        original_query: str,
        content: str,
        retrieval_top_k: int,
    ) -> Tuple[List[Any], List[Any]]:
        candidates_from_original = self._search_candidates(
            query_text=original_query,
            retrieval_top_k=retrieval_top_k,
        )
        extracted_answer = self.extract_answer_from_generation(generation=content)
        extracted_answer = self.split_on_keywords(text=extracted_answer)
        candidates_from_rewritten = self._search_candidates(
            query_text=extracted_answer,
            retrieval_top_k=retrieval_top_k,
        )
        return candidates_from_original, candidates_from_rewritten

    def _search_candidates(
        self,
        query_text: str,
        retrieval_top_k: int,
    ) -> List[Any]:
        query_embedding = self.embedding(
            input_text=query_text,
            is_query=True,
        )
        candidates = self.database.search(
            query_embedding=query_embedding,
            retrieval_top_k=retrieval_top_k,
        )
        candidates = sorted(
            candidates,
            key=lambda x: x[self.database.distance_column_name],
            reverse=True,
        )
        return [
            candidate[self.database.candidate_column_name] for candidate in candidates
        ]

    @staticmethod
    def _parse_ground_truth(
        gt: Union[str, List[str], np.ndarray, Any],
    ) -> Union[Any, List[Any]]:
        if isinstance(gt, str):
            try:
                return literal_eval(str(gt))
            except (ValueError, SyntaxError):
                return gt
        if isinstance(gt, np.ndarray):
            return gt.tolist()
        return gt

    @staticmethod
    def _flatten_ground_truth(
        parsed_gt: Union[Any, List[Any], np.ndarray],
    ) -> List[Any]:
        flat_gt: List[Any] = []
        items: List[Any] = [parsed_gt]
        index = 0
        while index < len(items):
            item = items[index]
            index += 1
            if isinstance(item, np.ndarray):
                items.extend(item.tolist())
                continue
            if isinstance(item, list):
                items.extend(item)
                continue
            flat_gt.append(item)
        return flat_gt

    @staticmethod
    def _build_ground_truth_lookup(
        flat_gt: List[Any],
    ) -> Tuple[Union[Set[Any], List[Any]], int]:
        if not flat_gt:
            return set(), 0

        try:
            gt_lookup: Union[Set[Any], List[Any]] = set(flat_gt)
            return gt_lookup, len(gt_lookup)
        except TypeError:
            unique_gt: List[Any] = []
            for item in flat_gt:
                is_new = True
                for seen in unique_gt:
                    if item == seen:
                        is_new = False
                        break
                if is_new:
                    unique_gt.append(item)
            return unique_gt, len(unique_gt)


class RetrievalHitReward(RetrievalBaseReward):
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
        database: FaissIndex,
        embedding: VllmEmbedding,
        retrieval_top_k: int,
        shaping_weight: float,
        rank_margin: int,
        stages: List[Dict[str, Any]],
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
            database=database,
            embedding=embedding,
        )

        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be >= 1")
        if shaping_weight < 0:
            raise ValueError("shaping_weight must be >= 0")
        if rank_margin < 0:
            raise ValueError("rank_margin must be >= 0")

        self.retrieval_top_k = retrieval_top_k
        self.shaping_weight = shaping_weight
        self.rank_margin = rank_margin

        self.stages = stages
        self._validate_stage_config()

    @property
    def name(self) -> str:
        stage_ks = ",".join(str(int(stage["k"])) for stage in self.stages)
        return f"retrieval_hit@{self.retrieval_top_k}_stages[{stage_ks}]_reward"

    def compute(
        self,
        completions: List[List[Dict[str, str]]],
        solution: List[Dict[str, Union[str, List[str]]]],
        reward_categories: List[str],
        **kwargs,
    ) -> List[Optional[float]]:
        rewards = []
        contents = self.get_contents_from_completions(completions=completions)
        for content, sol, category in zip(contents, solution, reward_categories):
            if not self.has_category_token(
                category=category,
                token="retrieval",
            ):
                rewards.append(None)
                continue

            original_query = sol["query"]
            gt = sol["candidate"]

            if not original_query:
                rewards.append(None)
                continue

            if not gt:
                rewards.append(None)
                continue

            self._ensure_ready()
            candidates_from_original, candidates_from_rewritten = (
                self._get_original_and_rewritten_candidates(
                    original_query=original_query,
                    content=content,
                    retrieval_top_k=self.retrieval_top_k,
                )
            )

            original_hit_location = 0
            rewritten_hit_location = 0

            parsed_gt = self._parse_ground_truth(gt=gt)
            flat_gt = self._flatten_ground_truth(parsed_gt=parsed_gt)

            if flat_gt:
                gt_lookup, _ = self._build_ground_truth_lookup(flat_gt=flat_gt)

                for idx, retrieved_candidate in enumerate(candidates_from_original):
                    if retrieved_candidate in gt_lookup:
                        original_hit_location = idx + 1
                        break

                for idx, retrieved_candidate in enumerate(candidates_from_rewritten):
                    if retrieved_candidate in gt_lookup:
                        rewritten_hit_location = idx + 1
                        break

            if original_hit_location > 0:
                original_rank_for_shaping = original_hit_location
            else:
                original_rank_for_shaping = self.retrieval_top_k + 1

            if rewritten_hit_location > 0:
                rewritten_rank_for_shaping = rewritten_hit_location
            else:
                rewritten_rank_for_shaping = self.retrieval_top_k + 1

            base = (
                math.log(original_rank_for_shaping)
                - math.log(rewritten_rank_for_shaping)
            ) / math.log(self.retrieval_top_k + 1)

            if (
                self.rank_margin > 0
                and abs(original_rank_for_shaping - rewritten_rank_for_shaping)
                <= self.rank_margin
            ):
                base = 0.0

            base *= self.shaping_weight

            bonus = 0.0
            penalty = 0.0

            best_bonus_stage = None
            for stage in self.stages:
                if rewritten_rank_for_shaping <= stage["k"]:
                    if best_bonus_stage is None or stage["k"] < best_bonus_stage["k"]:
                        best_bonus_stage = stage

            if best_bonus_stage is not None:
                bonus = best_bonus_stage["bonus"]

            best_drop_stage = None
            for stage in self.stages:
                if (
                    original_rank_for_shaping <= stage["k"]
                    and rewritten_rank_for_shaping > stage["k"]
                ):
                    if best_drop_stage is None or stage["k"] < best_drop_stage["k"]:
                        best_drop_stage = stage

            if best_drop_stage is not None:
                penalty = best_drop_stage["drop"]

            reward = base + bonus - penalty

            if reward > 1.0:
                reward = 1.0
            elif reward < -1.0:
                reward = -1.0

            rewards.append(float(reward))

        return rewards

    def _validate_stage_config(self) -> None:
        if (
            self.stages is None
            or not isinstance(self.stages, (list, ListConfig))
            or len(self.stages) == 0
        ):
            raise ValueError("stages must be a non-empty list of dicts")

        prev_k: Optional[int] = None
        prev_bonus: Optional[float] = None
        prev_drop: Optional[float] = None

        for i, stage in enumerate(self.stages):
            if not isinstance(stage, (dict, DictConfig)):
                raise ValueError(f"stages[{i}] must be a dict, got {type(stage)}")

            for key in ("k", "bonus", "drop"):
                if key not in stage:
                    raise ValueError(f"stages[{i}] missing required key '{key}'")

            k = stage["k"]
            bonus = stage["bonus"]
            drop = stage["drop"]

            if not isinstance(k, int):
                raise ValueError(f"stages[{i}]['k'] must be int, got {type(k)}")
            if k <= 0:
                raise ValueError(f"stages[{i}]['k'] must be >= 1, got {k}")
            if k > self.retrieval_top_k:
                raise ValueError(
                    f"stages[{i}]['k'] must be <= retrieval_top_k={self.retrieval_top_k}, got {k}"
                )

            try:
                bonus_f = float(bonus)
                drop_f = float(drop)
            except Exception:
                raise ValueError(f"stages[{i}] bonus/drop must be numeric")

            if bonus_f < 0:
                raise ValueError(f"stages[{i}]['bonus'] must be >= 0, got {bonus_f}")
            if drop_f < 0:
                raise ValueError(f"stages[{i}]['drop'] must be >= 0, got {drop_f}")

            if prev_k is not None and k > prev_k:
                raise ValueError(
                    f"Stage k order invalid at stages[{i}]: require non-increasing k "
                    f"(tighter later). Got prev_k={prev_k}, k={k}"
                )

            if prev_bonus is not None and bonus_f < prev_bonus:
                raise ValueError(
                    f"Stage bonus order invalid at stages[{i}]: require non-decreasing bonus "
                    f"(tighter later). Got prev_bonus={prev_bonus}, bonus={bonus_f}"
                )

            if prev_drop is not None and drop_f < prev_drop:
                raise ValueError(
                    f"Stage drop order invalid at stages[{i}]: require non-decreasing drop "
                    f"(tighter later). Got prev_drop={prev_drop}, drop={drop_f}"
                )

            prev_k = k
            prev_bonus = bonus_f
            prev_drop = drop_f


class RetrievalnDCGReward(RetrievalBaseReward):
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
        database: FaissIndex,
        embedding: VllmEmbedding,
        retrieval_top_k: Optional[int],
        reward_mode: str,
        ndcg_top_ks: List[int],
        alpha: float,
        weighting_mode: str,
        epsilon: float,
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
            database=database,
            embedding=embedding,
        )
        if reward_mode not in ["relative", "absolute"]:
            raise ValueError("reward_mode must be one of ['relative', 'absolute']")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        if weighting_mode not in ["small_k", "large_k"]:
            raise ValueError("weighting_mode must be one of ['small_k', 'large_k']")
        if reward_mode == "relative" and epsilon <= 0:
            raise ValueError("epsilon must be > 0 for relative reward_mode")

        self.reward_mode = reward_mode
        self.ndcg_top_ks = [int(k) for k in ndcg_top_ks]
        self.retrieval_top_k = self._resolve_retrieval_top_k(
            retrieval_top_k=retrieval_top_k,
        )
        self.alpha = alpha
        self.weighting_mode = weighting_mode
        self.epsilon = epsilon

        self._validate_ndcg_top_ks()
        self.ndcg_weights = self._build_ndcg_weights()

    @property
    def name(self) -> str:
        ks = ",".join(str(k) for k in self.ndcg_top_ks)
        return f"retrieval_ndcg@{ks}_{self.reward_mode}_{self.weighting_mode}_reward"

    def compute(
        self,
        completions: List[List[Dict[str, str]]],
        solution: List[Dict[str, Union[str, List[str]]]],
        reward_categories: List[str],
        **kwargs,
    ) -> List[Optional[float]]:
        rewards = []
        contents = self.get_contents_from_completions(completions=completions)
        for content, sol, category in zip(contents, solution, reward_categories):
            if not self.has_category_token(
                category=category,
                token="retrieval",
            ):
                rewards.append(None)
                continue

            original_query = sol["query"]
            gt = sol["candidate"]

            if not original_query:
                rewards.append(None)
                continue

            if not gt:
                rewards.append(None)
                continue

            self._ensure_ready()
            cached_original_ndcg = self._get_cached_original_ndcg(solution=sol)
            candidates_from_rewritten = self._get_rewritten_candidates(
                content=content,
                retrieval_top_k=self.retrieval_top_k,
            )
            candidates_from_original = None
            if self.reward_mode == "relative" and cached_original_ndcg is None:
                candidates_from_original = self._search_candidates(
                    query_text=original_query,
                    retrieval_top_k=self.retrieval_top_k,
                )

            parsed_gt = self._parse_ground_truth(gt=gt)
            flat_gt = self._flatten_ground_truth(parsed_gt=parsed_gt)

            gt_lookup, num_relevant = self._build_ground_truth_lookup(flat_gt=flat_gt)
            if num_relevant == 0:
                rewards.append(None)
                continue

            reward = 0.0
            for k, weight in zip(self.ndcg_top_ks, self.ndcg_weights):
                rewritten_ndcg = self._compute_ndcg(
                    ranked_candidates=candidates_from_rewritten,
                    gt_lookup=gt_lookup,
                    num_relevant=num_relevant,
                    top_k=k,
                )
                if self.reward_mode == "relative":
                    if cached_original_ndcg is not None:
                        original_ndcg = cached_original_ndcg[k]
                    else:
                        if candidates_from_original is None:
                            raise ValueError(
                                "candidates_from_original must be available when original_ndcg cache is missing"
                            )
                        original_ndcg = self._compute_ndcg(
                            ranked_candidates=candidates_from_original,
                            gt_lookup=gt_lookup,
                            num_relevant=num_relevant,
                            top_k=k,
                        )
                    reward_component = self._normalize_delta(
                        original_ndcg=original_ndcg,
                        rewritten_ndcg=rewritten_ndcg,
                        epsilon=self.epsilon,
                    )
                else:
                    reward_component = rewritten_ndcg
                reward += weight * reward_component

            if reward > 1.0:
                reward = 1.0
            elif reward < -1.0:
                reward = -1.0

            rewards.append(float(reward))

        return rewards

    def _get_cached_original_ndcg(
        self,
        solution: Dict[str, Any],
    ) -> Optional[Dict[int, float]]:
        original_ndcg = solution.get("original_ndcg")
        if not isinstance(original_ndcg, dict):
            return None

        cached_original_ndcg: Dict[int, float] = {}
        for k in self.ndcg_top_ks:
            key = str(k)
            if key not in original_ndcg:
                return None

            value = original_ndcg[key]
            if value is None:
                return None

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return None

            if not math.isfinite(numeric_value):
                return None

            cached_original_ndcg[k] = numeric_value

        return cached_original_ndcg

    def _get_rewritten_candidates(
        self,
        content: str,
        retrieval_top_k: int,
    ) -> List[Any]:
        extracted_answer = self.extract_answer_from_generation(generation=content)
        extracted_answer = self.split_on_keywords(text=extracted_answer)
        return self._search_candidates(
            query_text=extracted_answer,
            retrieval_top_k=retrieval_top_k,
        )

    def _validate_ndcg_top_ks(self) -> None:
        if (
            self.ndcg_top_ks is None
            or not isinstance(self.ndcg_top_ks, (list, ListConfig))
            or len(self.ndcg_top_ks) == 0
        ):
            raise ValueError("ndcg_top_ks must be a non-empty list of ints")

        prev_k: Optional[int] = None
        for i, k in enumerate(self.ndcg_top_ks):
            if not isinstance(k, int):
                raise ValueError(f"ndcg_top_ks[{i}] must be int, got {type(k)}")
            if k <= 0:
                raise ValueError(f"ndcg_top_ks[{i}] must be >= 1, got {k}")
            if k > self.retrieval_top_k:
                raise ValueError(
                    f"ndcg_top_ks[{i}] must be <= retrieval_top_k={self.retrieval_top_k}, got {k}"
                )
            if prev_k is not None and k <= prev_k:
                raise ValueError(
                    f"ndcg_top_ks must be strictly increasing. Got prev_k={prev_k}, k={k}"
                )
            prev_k = k

    def _resolve_retrieval_top_k(
        self,
        retrieval_top_k: Optional[int],
    ) -> int:
        if len(self.ndcg_top_ks) == 0:
            raise ValueError("ndcg_top_ks must be a non-empty list of ints")

        max_ndcg_top_k = max(self.ndcg_top_ks)
        if retrieval_top_k is None:
            return max_ndcg_top_k
        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be >= 1")
        if retrieval_top_k < max_ndcg_top_k:
            raise ValueError(
                "retrieval_top_k must be >= max(ndcg_top_ks). "
                f"Got retrieval_top_k={retrieval_top_k}, max_ndcg_top_k={max_ndcg_top_k}"
            )
        return retrieval_top_k

    def _build_ndcg_weights(self) -> List[float]:
        if self.weighting_mode == "small_k":
            raw_weights = [float(k) ** (-self.alpha) for k in self.ndcg_top_ks]
        else:
            raw_weights = [float(k) ** self.alpha for k in self.ndcg_top_ks]
        weight_sum = float(sum(raw_weights))
        if weight_sum <= 0:
            raise ValueError("invalid ndcg weights: sum must be > 0")
        return [weight / weight_sum for weight in raw_weights]

    @staticmethod
    def _normalize_delta(
        original_ndcg: float,
        rewritten_ndcg: float,
        epsilon: float,
    ) -> float:
        delta = rewritten_ndcg - original_ndcg
        if delta >= 0:
            denom = (1.0 - original_ndcg) + epsilon
        else:
            denom = original_ndcg + epsilon
        return delta / denom

    @staticmethod
    def _compute_ndcg(
        ranked_candidates: List[Any],
        gt_lookup: Union[Set[Any], List[Any]],
        num_relevant: int,
        top_k: int,
    ) -> float:
        if num_relevant <= 0:
            return 0.0

        limit = min(top_k, len(ranked_candidates))
        if limit <= 0:
            return 0.0

        dcg = 0.0
        for rank_index in range(limit):
            candidate = ranked_candidates[rank_index]
            if candidate in gt_lookup:
                dcg += 1.0 / math.log2(rank_index + 2)

        ideal_hits = min(num_relevant, limit)
        if ideal_hits <= 0:
            return 0.0

        idcg = 0.0
        for rank_index in range(ideal_hits):
            idcg += 1.0 / math.log2(rank_index + 2)

        if idcg <= 0:
            return 0.0

        return dcg / idcg
