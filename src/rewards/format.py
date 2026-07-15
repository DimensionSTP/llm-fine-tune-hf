from typing import Dict, List, Optional
import re

from .base import BaseReward


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
