from collections.abc import Iterator

import gradio as gr
from openai.types.chat import ChatCompletionMessageParam

from llm_chatbot.llm import ChatError, stream_reply


def _to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def respond(message: str, history: list) -> Iterator[str]:
    messages: list[ChatCompletionMessageParam] = []

    for turn in history:
        text = _to_text(turn["content"])
        if turn["role"] == "user":
            messages.append({"role": "user", "content": text})
        elif turn["role"] == "assistant":
            messages.append({"role": "assistant", "content": text})

    messages.append({"role": "user", "content": message})

    reply = ""
    try:
        for piece in stream_reply(messages):
            reply += piece
            yield reply
    except ChatError as error:
        note = f"**Error:** {error}"
        yield f"{reply}\n\n{note}" if reply else note


def build_app() -> gr.ChatInterface:
    return gr.ChatInterface(
        fn=respond,
        title="LLM Chatbot",
        description="A simple streaming chatbot built with Python.",
    )