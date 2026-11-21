from collections.abc import Iterator

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from llm_chatbot.config import get_settings

SYSTEM_PROMPT = "You are a helpful assistant. Keep answers clear and concise."


def stream_reply(messages: list[ChatCompletionMessageParam]) -> Iterator[str]:
    settings = get_settings()
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    full_messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages,
    ]

    stream = client.chat.completions.create(
        model=settings.model,
        messages=full_messages,
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.content
        if piece:
            yield piece