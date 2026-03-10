from typing import Dict, List, Tuple, Optional, Callable, Union, Any

from abc import ABC, abstractmethod
import re
import json
import unicodedata
import multiprocessing as mp
import contextlib
import io
import queue
import functools
import math

import numpy as np
from rouge_score import rouge_scorer
from ast import literal_eval

from omegaconf import DictConfig, ListConfig

from src.utils.reward_vector_store import FaissIndex
from src.utils.reward_embedding import VllmEmbedding


class BaseReward(ABC):
    def __init__(
        self,
        is_answer_tag: bool,
        think_start_token: str,
        think_end_token: str,
        answer_start_token: str,
        answer_end_token: str,
        eos_token: str,
        weight: float,
    ) -> None:
        self.is_answer_tag = is_answer_tag
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self.answer_start_token = answer_start_token
        self.answer_end_token = answer_end_token
        self.eos_token = eos_token
        self.weight = weight

    @property
    def name(self) -> str:
        return re.sub(
            r"(?<!^)(?=[A-Z])",
            "_",
            self.__class__.__name__,
        ).lower()

    def __call__(
        self,
        completions: List[List[Dict[str, str]]],
        solution: List[str],
        reward_categories: List[str],
        **kwargs,
    ) -> List[Optional[float]]:
        rewards = self.compute(
            completions=completions,
            solution=solution,
            reward_categories=reward_categories,
            **kwargs,
        )
        return [
            reward * self.weight if reward is not None else None for reward in rewards
        ]

    @abstractmethod
    def compute(
        self,
        completions: List[List[Dict[str, str]]],
        solution: List[str],
        reward_categories: List[str],
        **kwargs,
    ) -> List[Optional[float]]:
        pass

    @staticmethod
    def get_contents_from_completions(
        completions: List[List[Dict[str, str]]],
    ) -> List[str]:
        contents = [completion[0]["content"] for completion in completions]
        return contents

    def extract_answer_from_generation(
        self,
        generation: str,
    ) -> str:
        if not isinstance(generation, str):
            return ""

        if self.is_answer_tag:
            match = re.search(
                rf"{self.answer_start_token}(.*?){self.answer_end_token}",
                generation,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if match:
                return match.group(1).strip()
            return ""

        match = re.search(
            r"###\s*Start\s*\n(.*?)\n?###\s*End",
            generation,
            flags=re.DOTALL,
        )
        if match:
            return match.group(1).strip()

        match = re.search(
            r"<solution>(.*?)</solution>",
            generation,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        match = re.search(
            r"<answer>(.*?)</answer>",
            generation,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        match = re.search(
            r"###\s*Start\s*\n(.*)$",
            generation,
            flags=re.DOTALL,
        )
        if match:
            return match.group(1).strip()

        match = re.search(
            r"<solution>(.*)$",
            generation,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        match = re.search(
            r"<answer>(.*)$",
            generation,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        match = re.search(
            rf"{self.think_end_token}\s*(.*?)\s*(?:{self.eos_token}|$)",
            generation,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        return generation

    @staticmethod
    def split_on_keywords(text: str) -> str:
        if not isinstance(text, str):
            return ""
        pattern = r"answer\s*(?:is\s*:?|:)\s*"
        parts = re.split(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        if not parts:
            return str(text).strip()
        return parts[-1].strip()

    @staticmethod
    def strip_wrappers(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = re.sub(
            r"^[\s:=\-\(\)\[\]\{\}\|>]+",
            "",
            text,
        )
        text = re.sub(
            r"[\s:=\-\(\)\[\]\{\}\|<]+$",
            "",
            text,
        )
        text = re.sub(
            r"^\$+|\$+$",
            "",
            text,
        )
        while True:
            before = text
            text = re.sub(
                r"\\boxed\{([^{}]*)\}",
                r"\1",
                text,
            )
            text = re.sub(
                r"\\text\{([^{}]*)\}",
                r"\1",
                text,
            )
            if text == before:
                break
        text = text.replace(
            "\\(",
            "",
        ).replace(
            "\\)",
            "",
        )
        text = text.replace(
            "\\[",
            "",
        ).replace(
            "\\]",
            "",
        )
        text = text.replace(
            "\\",
            "",
        )
        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()
        return text

    @staticmethod
    def normalize_text(text: str) -> str:
        text = str(text).lower().strip()
        text = re.sub(
            r"[\s\.,;:!?\'\"]+",
            " ",
            text,
        )
        return text.strip()

    @staticmethod
    def has_category_token(
        category: Any,
        token: str,
    ) -> bool:
        if not isinstance(category, str):
            return False
        if not isinstance(token, str):
            return False

        token = token.lower().strip()
        if token == "":
            return False

        category_tokens = category.lower().split("_")
        return token in category_tokens


class RewardManager:
    def __init__(
        self,
        rewards: List[BaseReward],
    ) -> None:
        self.rewards = [reward for reward in rewards if reward.weight > 0]

    def get_reward_funcs(self) -> List[Callable]:
        funcs = []
        for reward in self.rewards:

            @functools.wraps(reward.__call__)
            def wrapper(
                *args,
                _reward=reward,
                **kwargs,
            ):
                return _reward(*args, **kwargs)

            wrapper.__name__ = reward.name
            funcs.append(wrapper)
        return funcs


class ThinkFormatReward(BaseReward):
    def __init__(
        self,
        is_answer_tag: bool,
        think_start_token: str,
        think_end_token: str,
        answer_start_token: str,
        answer_end_token: str,
        eos_token: str,
        weight: float,
        is_enable_thinking: bool,
    ) -> None:
        super().__init__(
            is_answer_tag=is_answer_tag,
            think_start_token=think_start_token,
            think_end_token=think_end_token,
            answer_start_token=answer_start_token,
            answer_end_token=answer_end_token,
            eos_token=eos_token,
            weight=weight,
        )
        self.is_enable_thinking = is_enable_thinking

    def compute(
        self,
        completions: List[List[Dict[str, str]]],
        solution: List[str],
        reward_categories: List[str],
        **kwargs,
    ) -> List[Optional[float]]:
        if not self.is_enable_thinking:
            return [None] * len(completions)

        pattern = rf"{self.think_start_token}(?!.*{self.think_start_token})(.*?){self.think_end_token}"
        contents = self.get_contents_from_completions(completions=completions)
        matches = [
            re.search(
                pattern,
                content,
                re.DOTALL | re.MULTILINE,
            )
            for content in contents
        ]
        return [1.0 if match else 0.0 for match in matches]


class AnswerFormatReward(BaseReward):
    def compute(
        self,
        completions: List[List[Dict[str, str]]],
        solution: List[str],
        reward_categories: List[str],
        **kwargs,
    ) -> List[Optional[float]]:
        if not self.is_answer_tag:
            return [None] * len(completions)

        pattern = rf"{self.answer_start_token}(?!.*{self.answer_start_token})(.*?){self.answer_end_token}"
        contents = self.get_contents_from_completions(completions=completions)
        matches = [
            re.search(
                pattern,
                content,
                re.DOTALL | re.MULTILINE,
            )
            for content in contents
        ]
        return [1.0 if match else 0.0 for match in matches]


class MatchReward(BaseReward):
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
            if category not in ["math", "choice"]:
                rewards.append(None)
                continue

            if not sol:
                rewards.append(None)
                continue

            extracted_answer = self.extract_answer_from_generation(generation=content)
            extracted_answer = self.split_on_keywords(text=extracted_answer)
            if extracted_answer == sol:
                rewards.append(1.0)
                continue

            clean_answer = self.strip_wrappers(text=extracted_answer)
            if clean_answer == sol:
                rewards.append(1.0)
                continue

            if self.normalize_text(text=clean_answer) == self.normalize_text(text=sol):
                rewards.append(1.0)
                continue

            answer_choice = self.extract_choice(text=clean_answer)
            solution_choice = self.extract_choice(text=sol)
            if answer_choice and solution_choice:
                if answer_choice == solution_choice:
                    rewards.append(1.0)
                    continue

            answer_number = self.extract_number(text=clean_answer)
            solution_number = self.extract_number(text=sol)
            if answer_number and solution_number:
                if answer_number == solution_number:
                    rewards.append(1.0)
                    continue

            rewards.append(0.0)

        return rewards

    @staticmethod
    def extract_choice(text: str) -> str:
        if not text:
            return ""
        match = re.match(
            r"^[\s:=\-\(\[\{]*([A-Ea-e])[\)\]\}\s\.,;:!?]*$",
            text,
        )
        if match:
            return match.group(1).upper()
        match = re.search(
            r"\b([A-Ea-e])\b",
            text,
        )
        return match.group(1).upper() if match else ""

    @staticmethod
    def extract_number(text: str) -> str:
        if not text:
            return ""
        match = re.search(
            r"-?\d+(?:\.\d+)?",
            text,
        )
        if not match:
            return ""
        number = match.group(0)
        if re.fullmatch(r"-?\d+", number):
            sign = "-" if number.startswith("-") else ""
            digits = number[1:] if sign else number
            digits = digits.lstrip("0") or "0"
            return f"{sign}{digits}"
        return number


class CodeExecutionReward(BaseReward):
    def __init__(
        self,
        is_answer_tag: bool,
        think_start_token: str,
        think_end_token: str,
        answer_start_token: str,
        answer_end_token: str,
        eos_token: str,
        weight: float,
        timeout: int,
    ) -> None:
        super().__init__(
            is_answer_tag=is_answer_tag,
            think_start_token=think_start_token,
            think_end_token=think_end_token,
            answer_start_token=answer_start_token,
            answer_end_token=answer_end_token,
            eos_token=eos_token,
            weight=weight,
        )
        self.timeout = timeout

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
            if category != "code":
                rewards.append(None)
                continue

            if not sol:
                rewards.append(None)
                continue

            extracted_answer = self.extract_answer_from_generation(generation=content)
            extracted_answer = self.split_on_keywords(text=extracted_answer)

            answer_code = self.parse_python_code(text=extracted_answer)
            if not answer_code:
                rewards.append(0.0)
                continue

            answer_result = self.execute_python_code(
                code=answer_code,
                timeout=self.timeout,
            )
            if answer_result["status"] != "success":
                rewards.append(0.0)
                continue

            solution_code = self.parse_python_code(text=sol)
            if not solution_code:
                rewards.append(None)
                continue

            solution_result = self.execute_python_code(
                code=solution_code,
                timeout=self.timeout,
            )
            if solution_result["status"] != "success":
                rewards.append(None)
                continue

            if answer_result["output"] == solution_result["output"]:
                rewards.append(1.0)
            else:
                rewards.append(0.5)

        return rewards

    @staticmethod
    def parse_python_code(text: str) -> str:
        if not isinstance(text, str):
            return ""
        match = re.search(
            r"```python\s*(.*?)\s*```",
            text,
            flags=re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return ""

    def execute_python_code(
        self,
        code: str,
        timeout: int,
    ) -> Dict[str, str]:
        if timeout <= 0:
            raise ValueError("timeout must be a positive integer")

        if not isinstance(code, str) or not code.strip():
            return {
                "status": "no_code",
                "output": "",
                "error": "Empty or invalid python snippet",
            }

        result_queue: mp.Queue = mp.Queue()
        process = mp.Process(
            target=self._execute_code_worker,
            args=(
                code,
                result_queue,
            ),
        )
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            result = {
                "status": "timeout",
                "output": "",
                "error": f"Execution exceeded {timeout} seconds",
            }
        else:
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                result = {
                    "status": "error",
                    "output": "",
                    "error": "Execution finished without returning a result",
                }

        result_queue.close()
        result_queue.join_thread()
        return result

    @staticmethod
    def _execute_code_worker(
        code: str,
        result_queue: mp.Queue,
    ) -> None:
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                exec(
                    code,
                    {
                        "__name__": "__main__",
                    },
                )
            result_queue.put(
                {
                    "status": "success",
                    "output": buffer.getvalue().strip(),
                    "error": "",
                }
            )
        except Exception as exc:
            result_queue.put(
                {
                    "status": "error",
                    "output": buffer.getvalue().strip(),
                    "error": str(exc),
                }
            )


class RougeReward(BaseReward):
    def __init__(
        self,
        is_answer_tag: bool,
        think_start_token: str,
        think_end_token: str,
        answer_start_token: str,
        answer_end_token: str,
        eos_token: str,
        weight: float,
        rouge_type: str,
    ) -> None:
        super().__init__(
            is_answer_tag=is_answer_tag,
            think_start_token=think_start_token,
            think_end_token=think_end_token,
            answer_start_token=answer_start_token,
            answer_end_token=answer_end_token,
            eos_token=eos_token,
            weight=weight,
        )
        self.rouge_type = rouge_type

        if self.rouge_type not in ["1", "2", "l"]:
            raise ValueError("rouge_type must be '1', '2', or 'l'")

        self.scorer = rouge_scorer.RougeScorer(
            [f"rouge{self.rouge_type.upper()}"],
            use_stemmer=True,
        )

    @property
    def name(self) -> str:
        return f"rouge_{self.rouge_type}_reward"

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
            if category != "rouge":
                rewards.append(None)
                continue

            if not sol:
                rewards.append(None)
                continue

            extracted_answer = self.extract_answer_from_generation(generation=content)
            extracted_answer = self.split_on_keywords(text=extracted_answer)

            clean_answer = self.strip_wrappers(text=extracted_answer)

            score = self.calculate_rouge_score(
                prediction=clean_answer,
                reference=sol,
            )
            rewards.append(score)
        return rewards

    def calculate_rouge_score(
        self,
        prediction: str,
        reference: str,
    ) -> float:
        scores = self.scorer.score(
            prediction=prediction,
            target=reference,
        )
        return scores[f"rouge{self.rouge_type.upper()}"].fmeasure


class EquationReward(BaseReward):
    def __init__(
        self,
        is_answer_tag: bool,
        think_start_token: str,
        think_end_token: str,
        answer_start_token: str,
        answer_end_token: str,
        eos_token: str,
        weight: float,
        equation_target_column_name: str,
        equation_numbers_column_name: str,
    ) -> None:
        super().__init__(
            is_answer_tag=is_answer_tag,
            think_start_token=think_start_token,
            think_end_token=think_end_token,
            answer_start_token=answer_start_token,
            answer_end_token=answer_end_token,
            eos_token=eos_token,
            weight=weight,
        )
        self.equation_target_column_name = equation_target_column_name
        self.equation_numbers_column_name = equation_numbers_column_name

    def compute(
        self,
        completions: List[List[Dict[str, str]]],
        solution: List[Dict[str, Any]],
        reward_categories: List[str],
        **kwargs,
    ) -> List[Optional[float]]:
        rewards = []
        contents = self.get_contents_from_completions(completions=completions)
        for content, sol, category in zip(contents, solution, reward_categories):
            if category != "equation":
                rewards.append(None)
                continue

            if not sol:
                rewards.append(None)
                continue

            extracted_answer = self.extract_answer_from_generation(generation=content)
            extracted_answer = self.split_on_keywords(text=extracted_answer)

            clean_answer = self.strip_wrappers(text=extracted_answer)

            reward = self.calculate_equation_reward(
                equation=clean_answer,
                solution=sol,
            )
            rewards.append(reward)
        return rewards

    def calculate_equation_reward(
        self,
        equation: str,
        solution: Dict[str, Any],
    ) -> float:
        target = solution[self.equation_target_column_name]
        numbers = solution[self.equation_numbers_column_name]
        try:
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]
            if sorted(used_numbers) != sorted(numbers):
                return 0.0

            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation):
                return 0.0

            result = eval(equation, {"__builtins__": None}, {})
            if abs(float(result) - float(target)) < 1e-5:
                return 1.0
            else:
                return 0.0
        except Exception:
            return 0.0


class RetrievalBaseReward(BaseReward):
    def __init__(
        self,
        is_answer_tag: bool,
        think_start_token: str,
        think_end_token: str,
        answer_start_token: str,
        answer_end_token: str,
        eos_token: str,
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
    ) -> Tuple[Union[set[Any], List[Any]], int]:
        if not flat_gt:
            return set(), 0

        try:
            gt_lookup: Union[set[Any], List[Any]] = set(flat_gt)
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
        weight: float,
        database: FaissIndex,
        embedding: VllmEmbedding,
        retrieval_top_k: Optional[int],
        reward_mode: str,
        ndcg_top_ks: List[int],
        alpha: float,
        epsilon: float,
    ) -> None:
        super().__init__(
            is_answer_tag=is_answer_tag,
            think_start_token=think_start_token,
            think_end_token=think_end_token,
            answer_start_token=answer_start_token,
            answer_end_token=answer_end_token,
            eos_token=eos_token,
            weight=weight,
            database=database,
            embedding=embedding,
        )
        if reward_mode not in ["relative", "absolute"]:
            raise ValueError("reward_mode must be one of ['relative', 'absolute']")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        if reward_mode == "relative" and epsilon <= 0:
            raise ValueError("epsilon must be > 0 for relative reward_mode")

        self.reward_mode = reward_mode
        self.ndcg_top_ks = [int(k) for k in ndcg_top_ks]
        self.retrieval_top_k = self._resolve_retrieval_top_k(
            retrieval_top_k=retrieval_top_k,
        )
        self.alpha = alpha
        self.epsilon = epsilon

        self._validate_ndcg_top_ks()
        self.ndcg_weights = self._build_ndcg_weights()

    @property
    def name(self) -> str:
        ks = ",".join(str(k) for k in self.ndcg_top_ks)
        return f"retrieval_ndcg@{ks}_{self.reward_mode}_reward"

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
        raw_weights = [float(k) ** (-self.alpha) for k in self.ndcg_top_ks]
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
        gt_lookup: Union[set[Any], List[Any]],
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


class SingleKVReward(BaseReward):
    def __init__(
        self,
        is_answer_tag: bool,
        think_start_token: str,
        think_end_token: str,
        answer_start_token: str,
        answer_end_token: str,
        eos_token: str,
        weight: float,
        json_parse_weight: float,
    ) -> None:
        super().__init__(
            is_answer_tag=is_answer_tag,
            think_start_token=think_start_token,
            think_end_token=think_end_token,
            answer_start_token=answer_start_token,
            answer_end_token=answer_end_token,
            eos_token=eos_token,
            weight=weight,
        )
        self.json_parse_weight = json_parse_weight

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
                token="kv",
            ):
                rewards.append(None)
                continue

            if not sol:
                rewards.append(None)
                continue

            extracted_answer = self.extract_answer_from_generation(generation=content)
            extracted_answer = self.split_on_keywords(text=extracted_answer)

            pred_json = self._try_parse_json(text=extracted_answer)
            if pred_json is None:
                rewards.append(0.0)
                continue

            gt_json = self._try_parse_json(text=sol)
            if gt_json is None:
                rewards.append(None)
                continue

            if self._contains_tables(
                obj=pred_json,
            ) or self._contains_tables(
                obj=gt_json,
            ):
                reward = self._compute_table_reward(
                    pred_json=pred_json,
                    gt_json=gt_json,
                )
                rewards.append(reward)
                continue

            pred_leaf = self._extract_last_leaf_value(node=pred_json)
            gt_leaf = self._extract_last_leaf_value(node=gt_json)

            if self._values_match(
                pred_leaf=pred_leaf,
                gt_leaf=gt_leaf,
            ):
                rewards.append(1.0)
            else:
                rewards.append(self.json_parse_weight)

        return rewards

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
    def _extract_last_leaf_value(node: Any) -> Optional[Any]:
        leaves: List[Any] = []

        def walk(obj: Any) -> None:
            if isinstance(obj, dict):
                for val in obj.values():
                    walk(val)
            elif isinstance(obj, list):
                if any(isinstance(item, (dict, list)) for item in obj):
                    for item in obj:
                        walk(item)
                else:
                    leaves.append(obj)
            else:
                leaves.append(obj)

        walk(node)
        if not leaves:
            return None

        leaf = leaves[-1]
        if isinstance(leaf, list):
            return leaf[-1] if leaf else ""
        return leaf

    def _values_match(
        self,
        pred_leaf: Any,
        gt_leaf: Any,
    ) -> bool:
        pred_clean, pred_coarse = self._normalize_leaf(value=pred_leaf)
        gt_clean, gt_coarse = self._normalize_leaf(value=gt_leaf)

        if pred_clean == ("",) and gt_clean == ("",):
            return True
        if pred_clean and gt_clean and pred_clean == gt_clean:
            return True
        if pred_coarse and gt_coarse and pred_coarse == gt_coarse:
            return True
        return False

    def _normalize_leaf(
        self,
        value: Any,
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        if value is None:
            return ("",), ("",)

        if isinstance(value, list):
            if not value:
                return ("",), ("",)
            cleaned = tuple(
                self._clean_text(text=str(item)) if item is not None else ""
                for item in value
            )
            coarse = tuple(
                self._coarse_normalize(text=str(item)) if item is not None else ""
                for item in value
            )
            return cleaned, coarse

        text = str(value)
        return (self._clean_text(text=text),), (self._coarse_normalize(text=text),)

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

    @staticmethod
    def _contains_tables(obj: Any) -> bool:
        return isinstance(obj, dict) and "tables" in obj

    def _compute_table_reward(
        self,
        pred_json: Any,
        gt_json: Any,
    ) -> float:
        pred_tables = pred_json.get("tables") if isinstance(pred_json, dict) else None
        gt_tables = gt_json.get("tables") if isinstance(gt_json, dict) else None

        if not isinstance(pred_tables, dict) or not isinstance(gt_tables, dict):
            return self.json_parse_weight

        total_cells = 0
        matched_cells = 0

        table_names = set(gt_tables.keys()) | set(pred_tables.keys())

        for table_name in table_names:
            gt_table = gt_tables.get(table_name, {})
            pred_table = pred_tables.get(table_name, {})

            gt_rows = gt_table.get("rows", []) if isinstance(gt_table, dict) else []
            pred_rows = (
                pred_table.get("rows", []) if isinstance(pred_table, dict) else []
            )

            max_rows = max(len(gt_rows), len(pred_rows))
            for row_idx in range(max_rows):
                gt_row = gt_rows[row_idx] if row_idx < len(gt_rows) else {}
                pred_row = pred_rows[row_idx] if row_idx < len(pred_rows) else {}

                gt_vals = self._row_values_from_row(row=gt_row)
                pred_vals = self._row_values_from_row(row=pred_row)
                max_cells = max(len(gt_vals), len(pred_vals))

                for cell_idx in range(max_cells):
                    total_cells += 1
                    gt_val = gt_vals[cell_idx] if cell_idx < len(gt_vals) else [""]
                    pred_val = (
                        pred_vals[cell_idx] if cell_idx < len(pred_vals) else [""]
                    )
                    if self._values_match(
                        pred_leaf=pred_val,
                        gt_leaf=gt_val,
                    ):
                        matched_cells += 1

        if total_cells == 0:
            return self.json_parse_weight

        cell_accuracy = matched_cells / total_cells
        return self.json_parse_weight + (1 - self.json_parse_weight) * cell_accuracy

    @staticmethod
    def _row_values_from_row(row: Any) -> List[Any]:
        if not isinstance(row, dict):
            return []
        return [val for _, val in row.items()]


class MultiKVReward(SingleKVReward):
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
                token="kv",
            ):
                rewards.append(None)
                continue

            if not sol:
                rewards.append(None)
                continue

            extracted_answer = self.extract_answer_from_generation(generation=content)
            extracted_answer = self.split_on_keywords(text=extracted_answer)

            pred_json = self._try_parse_json(text=extracted_answer)
            if pred_json is None:
                rewards.append(0.0)
                continue

            gt_json = self._try_parse_json(text=sol)
            if gt_json is None:
                rewards.append(None)
                continue

            kv_total, kv_matched = self._compute_kv_counts(
                pred_json=pred_json,
                gt_json=gt_json,
            )
            table_total, table_matched = self._compute_table_counts(
                pred_json=pred_json,
                gt_json=gt_json,
            )

            total_items = kv_total + table_total
            matched_items = kv_matched + table_matched

            if total_items == 0:
                rewards.append(self.json_parse_weight)
                continue

            accuracy = matched_items / float(total_items)
            reward = self.json_parse_weight + (1 - self.json_parse_weight) * accuracy
            rewards.append(reward)

        return rewards

    def _compute_kv_counts(
        self,
        pred_json: Any,
        gt_json: Any,
    ) -> tuple[int, int]:
        pred_leaves = self._extract_leaf_values_excluding_tables(node=pred_json)
        gt_leaves = self._extract_leaf_values_excluding_tables(node=gt_json)

        if not pred_leaves and not gt_leaves:
            return 0, 0

        matched = 0
        remaining_pred = list(pred_leaves)
        for gt_leaf in gt_leaves:
            matched_idx = self._find_match_index(
                candidates=remaining_pred,
                target=gt_leaf,
            )
            if matched_idx is not None:
                matched += 1
                remaining_pred.pop(matched_idx)

        total = len(gt_leaves) + (len(pred_leaves) - matched)
        return total, matched

    def _find_match_index(
        self,
        candidates: List[Tuple[Tuple[str, ...], Any]],
        target: Tuple[Tuple[str, ...], Any],
    ) -> Optional[int]:
        target_path, target_leaf = target
        for idx, candidate in enumerate(candidates):
            candidate_path, candidate_leaf = candidate
            if candidate_path != target_path:
                continue
            if self._values_match(
                pred_leaf=candidate_leaf,
                gt_leaf=target_leaf,
            ):
                return idx
        return None

    def _extract_leaf_values_excluding_tables(
        self,
        node: Any,
    ) -> List[Tuple[Tuple[str, ...], Any]]:
        leaves: List[Tuple[Tuple[str, ...], Any]] = []

        def walk(obj: Any, path: List[str]) -> None:
            if isinstance(obj, dict):
                for key, val in obj.items():
                    if key == "tables":
                        continue
                    walk(val, path + [key])
            elif isinstance(obj, list):
                if any(isinstance(item, (dict, list)) for item in obj):
                    for item in obj:
                        walk(item, path)
                else:
                    leaves.append((tuple(path), obj))
            else:
                leaves.append((tuple(path), obj))

        walk(node, [])
        return leaves

    def _compute_table_counts(
        self,
        pred_json: Any,
        gt_json: Any,
    ) -> tuple[int, int]:
        pred_tables = pred_json.get("tables") if isinstance(pred_json, dict) else None
        gt_tables = gt_json.get("tables") if isinstance(gt_json, dict) else None
        pred_tables = pred_tables if isinstance(pred_tables, dict) else {}
        gt_tables = gt_tables if isinstance(gt_tables, dict) else {}

        total_cells = 0
        matched_cells = 0

        table_names = set(gt_tables.keys()) | set(pred_tables.keys())

        for table_name in table_names:
            gt_table = gt_tables.get(table_name, {})
            pred_table = pred_tables.get(table_name, {})

            gt_rows = gt_table.get("rows", []) if isinstance(gt_table, dict) else []
            pred_rows = (
                pred_table.get("rows", []) if isinstance(pred_table, dict) else []
            )

            max_rows = max(len(gt_rows), len(pred_rows))
            for row_idx in range(max_rows):
                gt_row = gt_rows[row_idx] if row_idx < len(gt_rows) else {}
                pred_row = pred_rows[row_idx] if row_idx < len(pred_rows) else {}

                gt_vals = self._row_values_from_row(row=gt_row)
                pred_vals = self._row_values_from_row(row=pred_row)
                max_cells = max(len(gt_vals), len(pred_vals))

                for cell_idx in range(max_cells):
                    total_cells += 1
                    gt_val = gt_vals[cell_idx] if cell_idx < len(gt_vals) else [""]
                    pred_val = (
                        pred_vals[cell_idx] if cell_idx < len(pred_vals) else [""]
                    )
                    if self._values_match(
                        pred_leaf=pred_val,
                        gt_leaf=gt_val,
                    ):
                        matched_cells += 1

        return total_cells, matched_cells
