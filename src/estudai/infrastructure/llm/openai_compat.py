"""Adaptador para qualquer endpoint compatível com a API Chat Completions da OpenAI.

Cobre o modelo de fronteira (Muse Spark 1.3) e qualquer outro fornecedor que exponha essa
API (vLLM, OpenRouter, Together etc.), bastando trocar `base_url` e `model` no .env.
"""

import time

from estudai.ports.llm import LLMError, LLMRequest, LLMResponse, TransientLLMError, Usage


class OpenAICompatClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout_s: float = 180,
        json_mode: bool = True,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("instale o extra: uv sync --extra openai") from exc
        self._model = model
        self._json_mode = json_mode
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0)

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: LLMRequest) -> LLMResponse:
        import openai

        messages = [{"role": "system", "content": request.system}]
        messages += [{"role": m.role, "content": m.content} for m in request.messages]
        extra = {"response_format": {"type": "json_object"}} if request.json_schema and self._json_mode else {}
        if request.max_output_tokens:
            extra["max_tokens"] = request.max_output_tokens
        inicio = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._model, messages=messages, temperature=request.temperature, **extra
            )
        except (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ) as exc:
            raise TransientLLMError(str(exc)) from exc
        except openai.OpenAIError as exc:
            raise LLMError(str(exc)) from exc
        uso = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=self._model,
            usage=Usage(
                input_tokens=uso.prompt_tokens if uso else 0,
                output_tokens=uso.completion_tokens if uso else 0,
            ),
            latency_s=time.perf_counter() - inicio,
        )
