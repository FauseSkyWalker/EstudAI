"""Construção dos modelos de chat (LangChain) a partir de parâmetros primitivos.

Não depende de `estudai.settings`: quem traduz configuração em argumentos é o bootstrap.
"""

from collections.abc import Sequence
from typing import Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

# O Gemini expõe uma API compatível com a da OpenAI; assim um único cliente atende os dois tiers.
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def build_chat_model(
    *,
    provider: Literal["gemini", "openai_compat"],
    model: str,
    api_key: str,
    temperature: float,
    max_retries: int,
    timeout_s: float,
    base_url: str | None = None,
    callbacks: Sequence[BaseCallbackHandler] = (),
) -> BaseChatModel:
    if provider == "gemini":
        base_url = base_url or GEMINI_OPENAI_BASE_URL
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout_s,
        callbacks=list(callbacks),
    )
