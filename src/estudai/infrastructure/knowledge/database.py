"""Base de conhecimento persistente para o projeto. Permitindo busca semântica, híbrida e léxica.

Utilizamos o `lancedb` para busca e base de dados. O embedder também é opcional, permitindo o descarte
de modelos densos e utilizando apenas o BM25, que é léxico.
"""

import lancedb
import numpy as np

from estudai.domain.curriculum import Topico
from estudai.domain.material import TipoDocumento
from estudai.ports.knowledge import Document, SearchHit, SearchMode
from estudai.ports.llm import Embedder

DOCS_TABLE_NAME = "documents"


class KnowledgeDatabase:
    def __init__(self, db_uri: str, embedder: Embedder | None = None):
        "Loads a pre-built knowledge-base from a lancedb URI"
        self.db = lancedb.connect(db_uri)
        self.embedder = embedder

    @classmethod
    def build(
        cls, db_uri: str, documentos: list[Document], embedder: Embedder | None = None, **kw
    ) -> "KnowledgeDatabase":
        """Builds a knowledge base from documents and an optional embedder.

        Returns a ready-to-use `KnowledgeDatabase` class.
        """

        if embedder is not None:
            embeddings = embedder.embed([d.texto for d in documentos], query=False)
            for i, d in enumerate(documentos):
                d.embedding = embeddings[i]

        db = lancedb.connect(db_uri)
        db.create_table(DOCS_TABLE_NAME, data=[d.model_dump() for d in documentos], mode="overwrite")

        t = db.open_table(DOCS_TABLE_NAME)

        # Create BM25 index. Vector search is exhaustive (flat) for now.
        t.create_fts_index("texto", language="Portuguese")
        # TODO: criar índice ANN (IvfFlat/IvfHnswSq) quando o corpus crescer muito
        # e remover o bypass_vector_index nas buscas vetoriais

        return cls(db_uri, embedder)

    def __len__(self):
        t = self.db.open_table(DOCS_TABLE_NAME)

        return t.count_rows()

    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.TEXTUAL,
        k: int = 5,
        tipo: TipoDocumento | None = None,
        topico: Topico | None = None,
    ) -> list[SearchHit]:
        t = self.db.open_table(DOCS_TABLE_NAME)

        usar_vetor = self.embedder is not None and mode != SearchMode.TEXTUAL
        vetor = self.embedder.embed([query], query=True)[0] if usar_vetor else None
        effective_mode = SearchMode.TEXTUAL if vetor is None else mode

        if effective_mode == SearchMode.TEXTUAL:
            results = t.search(query, query_type="fts")
        else:
            if effective_mode == SearchMode.SEMANTICA:
                results = t.search(vetor, vector_column_name="embedding", query_type="vector").bypass_vector_index()
            else:
                results = (
                    t.search(query_type="hybrid", vector_column_name="embedding")
                    .vector(vetor)
                    .text(query)
                    .bypass_vector_index()
                )

        if tipo:
            results = results.where(f"tipo = '{tipo.value}'")
        if topico:
            results = results.where(f"topico = '{topico.value}'")

        rows = results.limit(k).to_arrow().to_pylist()
        if vetor is not None:
            _completar_embeddings(t, rows)

        return self._hits(rows, vetor)

    @staticmethod
    def _hits(rows: list[dict], vetor: list[float] | None) -> list[SearchHit]:
        """Converte as linhas em `SearchHit`s, mantendo a ordem do modo de busca."""
        if vetor is not None:
            scores = [_cosseno(vetor, r.get("embedding")) for r in rows]
        else:
            raw = [float(r.get("_score", 0.0)) for r in rows]
            best = max(raw, default=0.0)
            scores = [value / best if best > 0 else 0.0 for value in raw]
        return [
            SearchHit(
                document=Document(**{k: v for k, v in row.items() if not k.startswith("_")}),
                score=min(1.0, max(0.0, score)),
            )
            for row, score in zip(rows, scores, strict=True)
        ]

    def get(self, doc_id: str) -> Document | None:
        t = self.db.open_table(DOCS_TABLE_NAME)

        rows = t.search().where(f"id = '{doc_id}'").limit(1).to_arrow().to_pylist()
        if not rows:
            return None

        return Document(**{k: v for k, v in rows[0].items() if not k.startswith("_")})


def _completar_embeddings(tabela, rows: list[dict]) -> None:
    """A busca textual (FTS) do LanceDB devolve a coluna de embedding vazia; lê os vetores por id."""
    faltando = [r["id"] for r in rows if r.get("embedding") is None]
    if not faltando:
        return
    ids = ", ".join("'" + i.replace("'", "''") + "'" for i in faltando)
    lidos = tabela.search().where(f"id IN ({ids})").select(["id", "embedding"]).limit(len(faltando)).to_arrow()
    por_id = {r["id"]: r["embedding"] for r in lidos.to_pylist()}
    for r in rows:
        if r.get("embedding") is None:
            r["embedding"] = por_id.get(r["id"])


def _cosseno(a: list[float], b: list[float] | None) -> float:
    if b is None:
        return 0.0
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    normas = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va @ vb) / normas if normas else 0.0
