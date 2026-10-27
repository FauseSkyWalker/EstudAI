"""Adaptador Google Gemini (SDK `google-genai`): modelo econômico e embeddings."""

import time

from estudai.ports.llm import LLMError, LLMRequest, LLMResponse, TransientLLMError, Usage

_TRANSIENTES = {408, 429, 500, 502, 503, 504}


def _client(api_key: str, timeout_s: float):
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover
        raise ImportError("instale o extra: uv sync --extra gemini") from exc
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))


class GeminiClient:
    def __init__(self, model: str, api_key: str, *, timeout_s: float = 180) -> None:
        self._model = model
        self._client = _client(api_key, timeout_s)

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: LLMRequest) -> LLMResponse:
        from google.genai import errors, types

        contents = [
            types.Content(role="user" if m.role == "user" else "model", parts=[types.Part.from_text(text=m.content)])
            for m in request.messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=request.system,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            response_mime_type="application/json" if request.json_schema else None,
        )
        inicio = time.perf_counter()
        try:
            response = self._client.models.generate_content(model=self._model, contents=contents, config=config)
        except errors.APIError as exc:
            erro = TransientLLMError if exc.code in _TRANSIENTES else LLMError
            raise erro(f"Gemini {exc.code}: {exc.message}") from exc
        uso = response.usage_metadata
        return LLMResponse(
            text=response.text or "",
            model=self._model,
            usage=Usage(
                input_tokens=(uso.prompt_token_count or 0) if uso else 0,
                # tokens de raciocínio também são cobrados como saída
                output_tokens=((uso.candidates_token_count or 0) + (uso.thoughts_token_count or 0)) if uso else 0,
            ),
            latency_s=time.perf_counter() - inicio,
        )


class GeminiEmbedder:
    def __init__(self, model: str, api_key: str, *, batch_size: int = 100, timeout_s: float = 120) -> None:
        self._model = model
        self._batch = batch_size
        self._client = _client(api_key, timeout_s)

    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        from google.genai import types

        config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT")
        vetores: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            response = self._client.models.embed_content(
                model=self._model, contents=texts[i : i + self._batch], config=config
            )
            vetores += [list(e.values) for e in response.embeddings]
        return vetores
