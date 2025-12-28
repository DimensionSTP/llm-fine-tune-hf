from typing import Dict, List, Optional, Callable

from abc import ABC, abstractmethod
import re
import multiprocessing as mp
import contextlib
import io
import queue
import functools

from rouge_score import rouge_scorer


class BaseReward(ABC):
    def __init__(
        self,
        is_reasoning_model: bool,
        is_answer_tag: bool,
        think_start_token: str,
        think_end_token: str,
        answer_start_token: str,
        answer_end_token: str,
        eos_token: str,
        weight: float,
    ) -> None:
        self.is_reasoning_model = is_reasoning_model
        self.is_answer_tag = is_answer_tag
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self.answer_start_token = answer_start_token
        self.answer_end_token = answer_end_token
        self.eos_token = eos_token
        self.weight = weight

    @property
    def name(self) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", self.__class__.__name__).lower()

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

        else:
            if self.is_reasoning_model:
                match = re.search(
                    rf"{self.think_end_token}\s*(.*?)\s*(?:{self.eos_token}|$)",
                    generation,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                if match:
                    return match.group(1).strip()

                return ""

            else:
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


class ThinkFormatReward(BaseReward):
    def compute(
        self,
        completions: List[List[Dict[str, str]]],
        solution: List[str],
        reward_categories: List[str],
        **kwargs,
    ) -> List[Optional[float]]:
        if not self.is_reasoning_model:
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
        is_reasoning_model: bool,
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
            is_reasoning_model=is_reasoning_model,
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

            clean_answer = self.strip_wrappers(text=extracted_answer)

            answer_code = self.parse_python_code(text=clean_answer)
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
            if answer_result["status"] != "success":
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
        is_reasoning_model: bool,
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
            is_reasoning_model=is_reasoning_model,
            is_answer_tag=is_answer_tag,
            think_start_token=think_start_token,
            think_end_token=think_end_token,
            answer_start_token=answer_start_token,
            answer_end_token=answer_end_token,
            eos_token=eos_token,
            weight=weight,
        )
        self.rouge_type = rouge_type

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
        if self.rouge_type not in ["1", "2", "l"]:
            raise ValueError("rouge_type must be '1', '2', or 'l'")

        scorer = rouge_scorer.RougeScorer(
            [f"rouge{self.rouge_type.upper()}"],
            use_stemmer=True,
        )
        scores = scorer.score(reference, prediction)
        return scores[f"rouge{self.rouge_type.upper()}"].fmeasure


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
