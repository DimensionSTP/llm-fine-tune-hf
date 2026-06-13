from typing import Dict, List, Tuple, Set, Union, Optional, Callable, Any
from abc import ABC, abstractmethod
import re
import json
import unicodedata
import multiprocessing as mp
import contextlib
import io
import queue
import math

from omegaconf import DictConfig, ListConfig

import numpy as np
from rouge_score import rouge_scorer
from ast import literal_eval

from src.utils.reward_vector_store import FaissIndex
from src.utils.reward_embedding import VllmEmbedding


def format_reward_name_float(
    value: float,
) -> str:
    formatted = f"{value:g}"
    return formatted.replace(".", "p")


class BaseReward(ABC):
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
    ) -> None:
        self.is_answer_tag = is_answer_tag
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self.answer_start_token = answer_start_token
        self.answer_end_token = answer_end_token
        self.eos_token = eos_token
        self.extraction_profile = extraction_profile
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
        completions: List[Any],
    ) -> List[str]:
        contents = []
        for completion in completions:
            if isinstance(completion, str):
                contents.append(completion)
                continue
            if (
                isinstance(completion, list)
                and len(completion) > 0
                and isinstance(completion[0], dict)
                and "content" in completion[0]
            ):
                contents.append(completion[0]["content"])
                continue
            raise TypeError(
                f"Unsupported completion payload for reward extraction: {type(completion).__name__}"
            )
        return contents

    def extract_answer_from_generation(
        self,
        generation: str,
    ) -> str:
        if not isinstance(generation, str):
            return ""

        generation = self._normalize_generation_for_extraction(generation=generation)

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

    def _normalize_generation_for_extraction(
        self,
        generation: str,
    ) -> str:
        if self.extraction_profile == "default":
            return generation
        if self.extraction_profile == "gemma4":
            return self._normalize_gemma4_generation(generation=generation)
        raise ValueError(
            f"Unsupported reward extraction profile: {self.extraction_profile}"
        )

    @staticmethod
    def _normalize_gemma4_generation(
        generation: str,
    ) -> str:
        text = generation.strip()
        text = re.sub(
            r"<\|channel\>thought\b.*?<channel\|>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(
            r"^\s*<\|turn\>model\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"^\s*<\|channel\>[A-Za-z0-9_\-]+\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        for stop_token in ("<turn|>", "<eos>", "<|tool_response|>"):
            stop_index = text.find(stop_token)
            if stop_index != -1:
                text = text[:stop_index]
        return text.strip()

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
        return [RewardFunctionAdapter(reward=reward) for reward in self.rewards]


class NamespacedLogger:
    def __init__(
        self,
        reward_name: str,
        callback: Callable[[str, Any], None],
    ) -> None:
        self.reward_name = reward_name
        self.callback = callback

    def __call__(
        self,
        key: str,
        value: Any,
    ) -> None:
        self.callback(
            f"{self.reward_name}/{key}",
            value,
        )


class RewardFunctionAdapter:
    def __init__(
        self,
        reward: BaseReward,
    ) -> None:
        self.reward = reward
        self.__name__ = reward.name

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> List[Optional[float]]:
        patched_kwargs = dict(kwargs)
        log_extra = patched_kwargs.get("log_extra")
        log_metric = patched_kwargs.get("log_metric")

        if callable(log_extra):
            patched_kwargs["log_extra"] = NamespacedLogger(
                reward_name=self.reward.name,
                callback=log_extra,
            )

        if callable(log_metric):
            patched_kwargs["log_metric"] = NamespacedLogger(
                reward_name=self.reward.name,
                callback=log_metric,
            )

        rewards = self.reward(
            *args,
            **patched_kwargs,
        )
        self.log_reward_outputs(
            rewards=rewards,
            log_extra=log_extra if callable(log_extra) else None,
            log_metric=log_metric if callable(log_metric) else None,
        )
        return rewards

    def log_reward_outputs(
        self,
        rewards: List[Optional[float]],
        log_extra: Optional[Callable[[str, Any], None]],
        log_metric: Optional[Callable[[str, Any], None]],
    ) -> None:
        if log_extra is not None:
            log_extra(
                f"{self.reward.name}/reward",
                rewards,
            )

        if log_metric is None:
            return

        total_count = len(rewards)
        if total_count == 0:
            return

        valid_rewards = [reward for reward in rewards if reward is not None]
        log_metric(
            f"{self.reward.name}/coverage",
            len(valid_rewards) / total_count,
        )
        if len(valid_rewards) == 0:
            return

        mean_reward = sum(valid_rewards) / len(valid_rewards)
        log_metric(
            f"{self.reward.name}/mean",
            mean_reward,
        )


class ThinkFormatReward(BaseReward):
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
        is_enable_thinking: bool,
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
        incorrect_penalty: float,
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
        self.incorrect_penalty = incorrect_penalty

    @property
    def name(self) -> str:
        if self.incorrect_penalty <= 0.0:
            return super().name
        penalty = format_reward_name_float(value=self.incorrect_penalty)
        return f"{super().name}_neg{penalty}"

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

            rewards.append(-self.incorrect_penalty)

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
        extraction_profile: str,
        weight: float,
        timeout: int,
        wrong_output_penalty: float,
        non_executable_penalty: float,
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
        self.timeout = timeout
        self.wrong_output_penalty = wrong_output_penalty
        self.non_executable_penalty = non_executable_penalty

    @property
    def name(self) -> str:
        if self.wrong_output_penalty <= 0.0 and self.non_executable_penalty <= 0.0:
            return super().name

        wrong_penalty = format_reward_name_float(value=self.wrong_output_penalty)
        non_executable_penalty = format_reward_name_float(
            value=self.non_executable_penalty,
        )
        return f"{super().name}_negw{wrong_penalty}_negx{non_executable_penalty}"

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
                rewards.append(-self.non_executable_penalty)
                continue

            answer_result = self.execute_python_code(
                code=answer_code,
                timeout=self.timeout,
            )
            if answer_result["status"] != "success":
                rewards.append(-self.non_executable_penalty)
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
                rewards.append(0.5 - self.wrong_output_penalty)

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
        extraction_profile: str,
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
            extraction_profile=extraction_profile,
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
        extraction_profile: str,
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
            extraction_profile=extraction_profile,
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


class SingleKVReward(BaseReward):
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
        json_parse_weight: float,
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
        leaves = SingleKVReward._collect_leaf_values(node=node)
        if not leaves:
            return None

        leaf = leaves[-1]
        if isinstance(leaf, list):
            return leaf[-1] if leaf else ""
        return leaf

    @staticmethod
    def _collect_leaf_values(
        node: Any,
    ) -> List[Any]:
        leaves: List[Any] = []
        stack = [node]
        while stack:
            obj = stack.pop()
            if isinstance(obj, dict):
                stack.extend(reversed(list(obj.values())))
                continue
            if isinstance(obj, list):
                if any(isinstance(item, (dict, list)) for item in obj):
                    stack.extend(reversed(obj))
                else:
                    leaves.append(obj)
                continue
            leaves.append(obj)
        return leaves

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

            gt_rows = self._normalize_table_rows(
                rows=gt_table.get("rows", []) if isinstance(gt_table, dict) else []
            )
            pred_rows = self._normalize_table_rows(
                rows=pred_table.get("rows", []) if isinstance(pred_table, dict) else []
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
        return [row[key] for key in SingleKVReward._sorted_mapping_keys(mapping=row)]

    @staticmethod
    def _normalize_table_rows(
        rows: Any,
    ) -> List[Any]:
        if isinstance(rows, list):
            return rows
        if not isinstance(rows, dict):
            return []
        return [rows[key] for key in SingleKVReward._sorted_mapping_keys(mapping=rows)]

    @staticmethod
    def _sorted_mapping_keys(
        mapping: Dict[Any, Any],
    ) -> List[Any]:
        keys = list(mapping.keys())
        if all(SingleKVReward._is_int_like(value=key) for key in keys):
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
    ) -> Tuple[int, int]:
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
        stack: List[Tuple[Any, List[str]]] = [
            (
                node,
                [],
            )
        ]
        while stack:
            obj, path = stack.pop()
            if isinstance(obj, dict):
                for key, val in reversed(list(obj.items())):
                    if key == "tables":
                        continue
                    stack.append(
                        (
                            val,
                            path + [key],
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
                else:
                    leaves.append((tuple(path), obj))
                continue
            leaves.append((tuple(path), obj))
        return leaves

    def _compute_table_counts(
        self,
        pred_json: Any,
        gt_json: Any,
    ) -> Tuple[int, int]:
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

            gt_rows = self._normalize_table_rows(
                rows=gt_table.get("rows", []) if isinstance(gt_table, dict) else []
            )
            pred_rows = self._normalize_table_rows(
                rows=pred_table.get("rows", []) if isinstance(pred_table, dict) else []
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
        schema_keys: Optional[Dict[str, List[str]]] = None,
        status_values: Optional[Dict[str, List[str]]] = None,
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
        min_reward: float,
        max_reward: float,
        schema_keys: Optional[Dict[str, List[str]]] = None,
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
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.schema_keys = self._normalize_schema_keys(schema_keys=schema_keys)

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
        reward += self._compute_extra_target_penalty(
            prediction_item_map=prediction_item_map,
            gold_item_map=gold_item_map,
        )
        reward += self.wrong_selection_penalty * prediction_invalid_count
        return self._clip_reward(reward=reward)

    def _compute_target_quality(
        self,
        prediction_item_map: Dict[str, Dict[str, Any]],
        gold_item_map: Dict[str, Dict[str, Any]],
    ) -> float:
        if not gold_item_map:
            return 1.0 if not prediction_item_map else 0.0

        total_quality = 0.0
        for target_id, gold_item in gold_item_map.items():
            prediction_item = prediction_item_map.get(target_id)
            if prediction_item is None:
                continue
            total_quality += self._compute_item_quality(
                prediction_item=prediction_item,
                gold_item=gold_item,
            )

        return total_quality / len(gold_item_map)

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

        item_score = self.partial_match_weight * self._selection_f1(
            pred_selected_ids=prediction_ids,
            gold_selected_ids=gold_ids,
        )
        if prediction_ids - gold_ids:
            item_score += self.wrong_selection_penalty
        if len(prediction_ids) > len(gold_ids):
            item_score += self.over_selection_penalty

        return min(
            1.0,
            max(
                0.0,
                item_score / max_item_score,
            ),
        )

    def _compute_extra_target_penalty(
        self,
        prediction_item_map: Dict[str, Dict[str, Any]],
        gold_item_map: Dict[str, Dict[str, Any]],
    ) -> float:
        extra_target_count = len(
            set(prediction_item_map.keys()) - set(gold_item_map.keys())
        )
        return self.wrong_selection_penalty * extra_target_count

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
            _, has_duplicate_ids = selected_ids
            if has_duplicate_ids:
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
    ) -> Optional[Tuple[Set[str], bool]]:
        value = self._get_schema_value(
            payload=item,
            logical_key="selected_ids",
        )
        if not isinstance(value, list):
            return None
        normalized_ids: List[str] = []
        for selected_id in value:
            if isinstance(selected_id, (str, int)):
                normalized_id = str(selected_id).strip()
                if normalized_id:
                    normalized_ids.append(normalized_id)
                    continue
            return None
        selected_id_set = set(normalized_ids)
        return selected_id_set, len(selected_id_set) != len(normalized_ids)

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
        pred_selected_ids: Set[str],
        gold_selected_ids: Set[str],
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
