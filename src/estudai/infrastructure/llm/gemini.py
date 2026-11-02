"""Embeddings do Google Gemini (SDK `google-genai`).

Os modelos de chat do Gemini são acessados pelo LangChain via `factory.build_chat_model`.
"""

import re
import time

_ESPERA = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)


def _client(api_key: str, timeout_s: float):
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover
        raise ImportError("instale o extra: uv sync --extra gemini") from exc
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))


class GeminiEmbedder:
    """No plano gratuito, a cota de embeddings é de 100 textos por minuto (cada texto de um lote
    conta como uma requisição). Ao estourar a cota por minuto, espera o tempo que a API indica e
    continua; a cota diária esgotada não tem como esperar, então o erro sobe."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        batch_size: int = 100,
        timeout_s: float = 120,
        max_esperas: int = 10,
        sleep=time.sleep,
    ) -> None:
        self._model = model
        self._batch = batch_size
        self._client = _client(api_key, timeout_s)
        self._max_esperas = max_esperas
        self._sleep = sleep

    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        from google.genai import types

        config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT")
        vetores: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            lote = texts[i : i + self._batch]
            response = self._com_espera(lambda lote=lote: self._client.models.embed_content(
                model=self._model, contents=lote, config=config
            ))  # fmt: skip
            vetores += [list(e.values) for e in response.embeddings]
        return vetores

    def _com_espera(self, chamada):
        from google.genai import errors

        for tentativa in range(self._max_esperas + 1):
            try:
                return chamada()
            except errors.ClientError as exc:
                mensagem = str(exc)
                if exc.code != 429 or "per day" in mensagem.lower() or tentativa == self._max_esperas:
                    raise
                espera = float(m.group(1)) + 1 if (m := _ESPERA.search(mensagem)) else 60.0
                print(f"Cota de embeddings por minuto atingida; aguardando {espera:.0f}s...")
                self._sleep(espera)
        raise AssertionError("inalcançável")
