"""Base de conhecimento vetorial com busca semântica, textual (BM25) e híbrida.

Persistida em disco como `documentos.jsonl` + `embeddings.npy`. Para o volume do projeto
(milhares de trechos) a busca exaustiva em numpy é suficiente e evita um banco vetorial.
"""

from pathlib import Path

import numpy as np

from estudai.domain.curriculum import Topico
from estudai.domain.material import TipoDocumento
from estudai.infrastructure.knowledge.bm25 import BM25Index
from estudai.ports.knowledge import Document, SearchHit, SearchMode
from estudai.ports.llm import Embedder

_DOCS = "documentos.jsonl"
_EMB = "embeddings.npy"


class HybridKnowledgeBase:
    def __init__(self, documentos: list[Document], embeddings: np.ndarray, embedder: Embedder, *, alpha: float = 0.5):
        if len(documentos) != len(embeddings):
            raise ValueError("documentos e embeddings com tamanhos diferentes")
        self._docs = documentos
        self._por_id = {d.id: d for d in documentos}
        self._emb = _normalizar_linhas(np.asarray(embeddings, dtype=np.float32))
        self._embedder = embedder
        self._alpha = alpha  # peso da busca semântica na híbrida
        self._bm25 = BM25Index([d.texto for d in documentos])

    @classmethod
    def build(cls, documentos: list[Document], embedder: Embedder, **kw) -> "HybridKnowledgeBase":
        embeddings = np.array(embedder.embed([d.texto for d in documentos]), dtype=np.float32)
        return cls(documentos, embeddings.reshape(len(documentos), -1), embedder, **kw)

    def save(self, diretorio: Path) -> None:
        diretorio.mkdir(parents=True, exist_ok=True)
        with (diretorio / _DOCS).open("w", encoding="utf-8") as f:
            for d in self._docs:
                f.write(d.model_dump_json() + "\n")
        np.save(diretorio / _EMB, self._emb)

    @classmethod
    def load(cls, diretorio: Path, embedder: Embedder, **kw) -> "HybridKnowledgeBase":
        if not (diretorio / _DOCS).exists():
            raise FileNotFoundError(f"índice não encontrado em {diretorio}; rode `estudai build-index`")
        with (diretorio / _DOCS).open(encoding="utf-8") as f:
            documentos = [Document.model_validate_json(linha) for linha in f if linha.strip()]
        return cls(documentos, np.load(diretorio / _EMB), embedder, **kw)

    def __len__(self) -> int:
        return len(self._docs)

    def get(self, doc_id: str) -> Document | None:
        return self._por_id.get(doc_id)

    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.HIBRIDA,
        k: int = 5,
        tipo: TipoDocumento | None = None,
        topico: Topico | None = None,
    ) -> list[SearchHit]:
        if not self._docs:
            return []
        mascara = np.array(
            [(tipo is None or d.tipo == tipo) and (topico is None or d.topico in (None, topico)) for d in self._docs]
        )
        if not mascara.any():
            return []
        scores = self._scores(query, mode, mascara)
        scores[~mascara] = -1.0
        ordem = np.argsort(-scores)[:k]
        return [SearchHit(document=self._docs[i], score=float(scores[i])) for i in ordem if scores[i] > 0]

    def _scores(self, query: str, mode: SearchMode, mascara: np.ndarray) -> np.ndarray:
        semantico = textual = None
        if mode in (SearchMode.SEMANTICA, SearchMode.HIBRIDA):
            q = _normalizar_linhas(np.array(self._embedder.embed([query], query=True), dtype=np.float32))[0]
            semantico = np.clip(self._emb @ q, 0.0, 1.0)
        if mode in (SearchMode.TEXTUAL, SearchMode.HIBRIDA):
            bruto = self._bm25.scores(query)
            maximo = bruto[mascara].max() if mascara.any() else 0.0
            textual = bruto / maximo if maximo > 0 else bruto
        if mode == SearchMode.SEMANTICA:
            return semantico
        if mode == SearchMode.TEXTUAL:
            return textual
        return self._alpha * semantico + (1 - self._alpha) * textual


def _normalizar_linhas(m: np.ndarray) -> np.ndarray:
    normas = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(normas == 0, 1, normas)
