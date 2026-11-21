from collections.abc import Iterator

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from llm_chatbot.config import ConfigError, get_settings

SYSTEM_PROMPT = "You are a helpful assistant. Keep answers clear and concise."

STATUS_MESSAGES = {
    400: "The request was rejected as invalid.",
    401: "The API key was rejected. Check LLM_API_KEY in .env.",
    404: "The model or URL was not found. Check LLM_MODEL and LLM_BASE_URL in .env.",
    429: "Rate limit or quota reached. Wait a minute, or check your plan and billing.",
    500: "The provider had an internal error. Try again in a moment.",
    503: "The model service is overloaded or unavailable. Try again in a minute.",
}


class ChatError(Exception):
    """An error with a readable message, meant to be shown in the chat."""


def _explain(error: Exception) -> str:
    if isinstance(error, ConfigError):
        return f"Setup problem: {error}"
    if isinstance(error, openai.APITimeoutError):
        return "The request timed out. Please try again."
    if isinstance(error, openai.APIConnectionError):
        return "Could not reach the model. Check your internet connection."
    if isinstance(error, openai.APIStatusError):
        text = STATUS_MESSAGES.get(error.status_code, "The model returned an error.")
        return f"{text} (error {error.status_code})"
    return f"Something went wrong. ({type(error).__name__})"


def stream_reply(messages: list[ChatCompletionMessageParam]) -> Iterator[str]:
    try:
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
    except Exception as error:
        raise ChatError(_explain(error)) from error