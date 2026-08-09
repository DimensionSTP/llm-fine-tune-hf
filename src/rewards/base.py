from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import re


def format_reward_name_float(
    value: float,
) -> str:
    formatted = f"{value:g}"
    return formatted.replace(
        ".",
        "p",
    )


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

    @property
    def name(self) -> str:
        return re.sub(
            r"(?<!^)(?=[A-Z])",
            "_",
            self.__class__.__name__,
        ).lower()

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
