"""Embeddings do Google Gemini (SDK `google-genai`).

Os modelos de chat do Gemini são acessados pelo LangChain via `factory.build_chat_model`.
"""


def _client(api_key: str, timeout_s: float):
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover
        raise ImportError("instale o extra: uv sync --extra gemini") from exc
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))


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
