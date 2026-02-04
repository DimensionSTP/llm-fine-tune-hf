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
            if category != "math" and category != "choice":
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


class RetrievalHitReward(BaseReward):
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
        margin: float,
        tau: float,
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
        self.margin = margin
        self.tau = tau

    @property
    def name(self) -> str:
        return f"retrieval_hit@{self.database.retrieval_top_k}_reward"

    def _ensure_ready(self) -> None:
        if not self._database_loaded:
            self.database.load()
            self._database_loaded = True

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
            if category != "retrieval_hit":
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

            original_query_embedding = self.embedding(
                input_text=original_query,
                is_query=True,
            )
            candidates_from_original = self.database.search(
                query_embedding=original_query_embedding
            )
            candidates_from_original = sorted(
                candidates_from_original,
                key=lambda x: x[self.database.distance_column_name],
                reverse=True,
            )
            candidates_from_original = [
                candidate[self.database.candidate_column_name]
                for candidate in candidates_from_original
            ]

            extracted_answer = self.extract_answer_from_generation(generation=content)
            extracted_answer = self.split_on_keywords(text=extracted_answer)

            rewritten_query_embedding = self.embedding(
                input_text=extracted_answer,
                is_query=True,
            )
            candidates_from_rewritten = self.database.search(
                query_embedding=rewritten_query_embedding
            )
            candidates_from_rewritten = sorted(
                candidates_from_rewritten,
                key=lambda x: x[self.database.distance_column_name],
                reverse=True,
            )
            candidates_from_rewritten = [
                candidate[self.database.candidate_column_name]
                for candidate in candidates_from_rewritten
            ]

            original_hit_location = 0
            rewritten_hit_location = 0

            if isinstance(gt, str):
                try:
                    parsed_gt = literal_eval(str(gt))
                except (ValueError, SyntaxError):
                    parsed_gt = gt
            elif isinstance(gt, np.ndarray):
                parsed_gt = gt.tolist()
            else:
                parsed_gt = gt

            if isinstance(parsed_gt, list):
                gt_set = set(parsed_gt)

                for idx, retrieved_candidate in enumerate(candidates_from_original):
                    if retrieved_candidate in gt_set:
                        original_hit_location = idx + 1
                        break

                for idx, retrieved_candidate in enumerate(candidates_from_rewritten):
                    if retrieved_candidate in gt_set:
                        rewritten_hit_location = idx + 1
                        break
            else:
                if parsed_gt in candidates_from_original:
                    original_hit_location = (
                        candidates_from_original.index(parsed_gt) + 1
                    )
                if parsed_gt in candidates_from_rewritten:
                    rewritten_hit_location = (
                        candidates_from_rewritten.index(parsed_gt) + 1
                    )

            original_score = (
                0.0
                if original_hit_location == 0
                else 1.0 / math.log2(original_hit_location + 1)
            )
            rewritten_score = (
                0.0
                if rewritten_hit_location == 0
                else 1.0 / math.log2(rewritten_hit_location + 1)
            )

            delta = rewritten_score - original_score
            if delta < self.margin:
                reward = 0.0
            else:
                reward = math.tanh(delta / self.tau)
            rewards.append(reward)

        return rewards


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
            if category != "vlm_single_kv":
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

        if pred_clean == "" and gt_clean == "":
            return True
        if pred_clean and gt_clean and pred_clean == gt_clean:
            return True
        if pred_coarse and gt_coarse and pred_coarse == gt_coarse:
            return True
        return False

    def _normalize_leaf(
        self,
        value: Any,
    ) -> Tuple[str, str]:
        if isinstance(value, list):
            value = value[-1] if value else ""
        if value is None:
            return "", ""
        text = str(value)
        return (
            self._clean_text(text=text),
            self._coarse_normalize(text=text),
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
