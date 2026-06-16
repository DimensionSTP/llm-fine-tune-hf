from typing import Dict, Set, Iterator, Any

from transformers.utils.chat_template_utils import _get_template_variables


def build_enable_thinking_kwargs(
    data_encoder: object,
    is_enable_thinking: bool,
) -> Dict[str, Any]:
    return filter_chat_template_kwargs(
        data_encoder=data_encoder,
        chat_template_kwargs={
            "enable_thinking": is_enable_thinking,
        },
    )


def filter_chat_template_kwargs(
    data_encoder: object,
    chat_template_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    supported_arguments = _get_chat_template_variables(data_encoder=data_encoder)
    return {
        key: value
        for key, value in chat_template_kwargs.items()
        if key in supported_arguments
    }


def _get_chat_template_variables(
    data_encoder: object,
) -> Set[str]:
    chat_template = getattr(
        data_encoder,
        "chat_template",
        None,
    )
    if chat_template is None:
        tokenizer = getattr(
            data_encoder,
            "tokenizer",
            None,
        )
        chat_template = getattr(
            tokenizer,
            "chat_template",
            None,
        )

    variables = set()
    for template in _iter_chat_templates(chat_template):
        variables.update(_get_template_variables(template))
    return variables


def _iter_chat_templates(chat_template: object) -> Iterator[str]:
    if isinstance(chat_template, str):
        yield chat_template
        return

    if isinstance(chat_template, dict):
        default_template = chat_template.get("default")
        if isinstance(default_template, str):
            yield default_template
            return

        for template in chat_template.values():
            if isinstance(template, str):
                yield template
