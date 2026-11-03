"""Construção dos modelos de chat (LangChain) a partir de parâmetros primitivos.

Não depende de `estudai.settings`: quem traduz configuração em argumentos é o bootstrap.
"""

from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_openai import ChatOpenAI

# Gemini e OpenRouter expõem APIs compatíveis com a da OpenAI; um único cliente atende todos.
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

Provider = Literal["gemini", "openai_compat", "openrouter", "claude_cli"]
StructuredMethod = Literal["json_schema", "function_calling", "json_mode"]


class ChatOpenAICompativel(ChatOpenAI):
    """`ChatOpenAI` cujo método de saída estruturada vem da configuração do modelo.

    Os agentes chamam `with_structured_output(schema)` sem saber o provedor; aqui o método
    padrão passa a ser o configurado, já que nem todo endpoint compatível aceita `json_schema`.
    """

    structured_output_method: StructuredMethod = "json_schema"

    def with_structured_output(self, schema=None, *, method: StructuredMethod | None = None, **kwargs: Any):
        return super().with_structured_output(schema, method=method or self.structured_output_method, **kwargs)


def build_chat_model(
    *,
    provider: Provider,
    model: str,
    api_key: str | None,
    temperature: float,
    max_retries: int,
    timeout_s: float,
    base_url: str | None = None,
    effort: str | None = None,
    structured_output_method: StructuredMethod = "json_schema",
    extra_body: dict[str, Any] | None = None,
    callbacks: Sequence[BaseCallbackHandler] = (),
    rate_limiter: BaseRateLimiter | None = None,
) -> BaseChatModel:
    """`rate_limiter` deve ser compartilhado por todos os modelos que consomem a mesma cota
    (no Gemini, a cota é por modelo e por projeto, não por instância)."""
    if provider == "claude_cli":
        from estudai.infrastructure.llm.claude_cli import ClaudeCLIChatModel

        # o CLI usa a própria autenticação e não aceita temperature
        return ClaudeCLIChatModel(
            model=model,
            timeout_s=timeout_s,
            max_retries=max_retries,
            effort=effort,
            callbacks=list(callbacks),
            rate_limiter=rate_limiter,
        )

    extras: dict[str, Any] = {}
    body = dict(extra_body or {})
    if provider == "gemini":
        base_url = base_url or GEMINI_OPENAI_BASE_URL
    elif provider == "openrouter":
        base_url = base_url or OPENROUTER_BASE_URL
        # cabeçalhos opcionais de identificação do app no OpenRouter
        extras["default_headers"] = {"HTTP-Referer": "https://github.com/FauseSkyWalker/EstudAI", "X-Title": "EstudAI"}
        # pede o custo real da chamada em `usage.cost` (lido pelo UsageLedgerCallback)
        body.setdefault("usage", {"include": True})
    if body:
        extras["extra_body"] = body
    return ChatOpenAICompativel(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout_s,
        callbacks=list(callbacks),
        rate_limiter=rate_limiter,
        structured_output_method=structured_output_method,
        **extras,
    )
