from typing import Dict, List, Optional, Callable

import re
import multiprocessing as mp
import contextlib
import io
import queue

from rouge_score import rouge_scorer


def extract_answer_from_generation(generation: str) -> str:
    if not isinstance(generation, str):
        return ""

    match = re.search(
        r"</think>\s*(.*?)\s*(?:<\|im_end\|>|$)",
        generation,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

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

    return generation


def split_on_keywords(text: str) -> str:
    if not isinstance(text, str):
        return ""
    pattern = (
        r"(?:the\s*(?:final|best)\s*)?answer\s*(?:is|:)\s*|"
        r"best\s*answer\s*(?:is|:)\s*|"
        r"answer\s*(?:is|:)\s*"
    )
    parts = re.split(
        pattern,
        text,
        flags=re.IGNORECASE,
    )
    if not parts:
        return str(text).strip()
    return parts[-1].strip()


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


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(
        r"[\s\.,;:!?\'\"]+",
        " ",
        text,
    )
    return text.strip()


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


def execute_python_code(
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
        target=_execute_code_worker,
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


def calculate_rouge_score(
    prediction: str,
    reference: str,
    rouge_type: str = "l",
) -> float:
    if rouge_type not in ["1", "2", "l"]:
        raise ValueError("rouge_type must be '1', '2', or 'l'")

    scorer = rouge_scorer.RougeScorer(
        [f"rouge{rouge_type.upper()}"],
        use_stemmer=True,
    )
    scores = scorer.score(reference, prediction)
    return scores[f"rouge{rouge_type.upper()}"].fmeasure


def _get_completion_content(
    completion: List[Dict[str, str]],
) -> str:
    if not completion:
        return ""
    return completion[-1].get("content", "")


def formatting_reward_func(
    completions: List[List[Dict[str, str]]],
    **kwargs,
) -> List[float]:
    rewards = []
    for completion in completions:
        content = _get_completion_content(completion=completion)

        match_no_thinking = re.search(
            r"<think>\s*</think>",
            content,
        )
        has_eos_token = "<|im_end|>" in content

        if match_no_thinking:
            rewards.append(1.0 if has_eos_token else 0.5)
            continue

        if "<think>" in content and "</think>" in content:
            rewards.append(0.5 if has_eos_token else 0.25)
            continue

        rewards.append(0.0)

    return rewards


def exact_match_reward_func(
    completions: List[List[Dict[str, str]]],
    solution: List[str],
    reward_categories: List[str],
    **kwargs,
) -> List[Optional[float]]:
    rewards = []
    for completion, sol, category in zip(completions, solution, reward_categories):
        if category != "exact_match":
            rewards.append(None)
            continue

        if sol is None:
            rewards.append(0.0)
            continue

        content = _get_completion_content(completion=completion)

        clean_completion = content.strip()
        clean_answer = sol.strip()

        if clean_completion == clean_answer:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def solution_reward_func(
    completions: List[List[Dict[str, str]]],
    solution: List[str],
    reward_categories: List[str],
    **kwargs,
) -> List[Optional[float]]:
    rewards = []
    for completion, sol, category in zip(completions, solution, reward_categories):
        if category != "math" and category != "choice":
            rewards.append(None)
            continue

        if not sol:
            rewards.append(0.0)
            continue

        content = _get_completion_content(completion=completion)

        extracted = extract_answer_from_generation(generation=content)
        if not extracted:
            extracted = split_on_keywords(text=content)

        if not extracted:
            rewards.append(0.0)
            continue

        clean_extracted = strip_wrappers(text=extracted)
        clean_answer = strip_wrappers(text=sol)

        if normalize_text(text=clean_extracted) == normalize_text(text=clean_answer):
            rewards.append(1.0)
            continue

        extracted_number = extract_number(text=clean_extracted)
        answer_number = extract_number(text=clean_answer)
        if extracted_number and answer_number:
            if extracted_number == answer_number:
                rewards.append(1.0)
                continue

        extracted_choice = extract_choice(text=clean_extracted)
        answer_choice = extract_choice(text=clean_answer)
        if extracted_choice and answer_choice:
            if extracted_choice == answer_choice:
                rewards.append(1.0)
                continue

        rewards.append(0.0)

    return rewards


def xml_structure_reward_func(
    completions: List[List[Dict[str, str]]],
    reward_categories: List[str],
    **kwargs,
) -> List[Optional[float]]:
    rewards = []
    for completion, category in zip(completions, reward_categories):
        if category != "xml":
            rewards.append(None)
            continue

        content = _get_completion_content(completion=completion)

        if "<solution>" in content and "</solution>" in content:
            rewards.append(0.1)
            continue

        rewards.append(0.0)
    return rewards


def rouge_reward_func(
    completions: List[List[Dict[str, str]]],
    solution: List[str],
    reward_categories: List[str],
    **kwargs,
) -> List[Optional[float]]:
    rewards = []
    for completion, sol, category in zip(completions, solution, reward_categories):
        if category != "rouge":
            rewards.append(None)
            continue

        if not sol:
            rewards.append(0.0)
            continue

        content = _get_completion_content(completion=completion)
        extracted = extract_answer_from_generation(generation=content)
        if not extracted:
            extracted = split_on_keywords(text=content)

        score = calculate_rouge_score(
            prediction=extracted,
            reference=sol,
            rouge_type="l",
        )
        rewards.append(score)
    return rewards


def code_execution_reward_func(
    completions: List[List[Dict[str, str]]],
    solution: List[str],
    reward_categories: List[str],
    **kwargs,
) -> List[Optional[float]]:
    rewards = []
    for completion, sol, category in zip(completions, solution, reward_categories):
        if category != "code":
            rewards.append(None)
            continue

        content = _get_completion_content(completion=completion)
        code = parse_python_code(text=content)
        if not code:
            rewards.append(0.0)
            continue

        result = execute_python_code(
            code=code,
            timeout=2,
        )
        if result["status"] != "success":
            rewards.append(0.0)
            continue

        if not sol:
            rewards.append(0.5)
            continue

        output = result["output"]
        clean_output = strip_wrappers(text=output)
        clean_sol = strip_wrappers(text=sol)

        is_correct = False
        if normalize_text(text=clean_output) == normalize_text(text=clean_sol):
            is_correct = True
        else:
            extracted_number = extract_number(text=clean_output)
            answer_number = extract_number(text=clean_sol)
            if extracted_number and answer_number:
                if extracted_number == answer_number:
                    is_correct = True

            if not is_correct:
                extracted_choice = extract_choice(text=clean_output)
                answer_choice = extract_choice(text=clean_sol)
                if extracted_choice and answer_choice:
                    if extracted_choice == answer_choice:
                        is_correct = True

        if is_correct:
            rewards.append(1.0)
        else:
            rewards.append(0.5)

    return rewards


def get_grpo_reward_functions() -> List[Callable]:
    return [
        formatting_reward_func,
        exact_match_reward_func,
        solution_reward_func,
        xml_structure_reward_func,
        rouge_reward_func,
        code_execution_reward_func,
    ]
