from typing import Dict, Iterator


def build_enable_thinking_kwargs(
    data_encoder: object,
    is_enable_thinking: bool,
) -> Dict[str, bool]:
    if _chat_template_supports_argument(
        data_encoder=data_encoder,
        argument_name="enable_thinking",
    ):
        return {
            "enable_thinking": is_enable_thinking,
        }
    return {}


def _chat_template_supports_argument(
    data_encoder: object,
    argument_name: str,
) -> bool:
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

    return any(
        argument_name in template for template in _iter_chat_templates(chat_template)
    )


def _iter_chat_templates(chat_template: object) -> Iterator[str]:
    if isinstance(chat_template, str):
        yield chat_template
        return

    if isinstance(chat_template, dict):
        for template in chat_template.values():
            if isinstance(template, str):
                yield template
