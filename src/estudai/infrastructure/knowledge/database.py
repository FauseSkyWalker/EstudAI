"""Base de conhecimento persistente para o projeto. Permitindo busca semântica, híbrida e léxica.

Utilizamos o `lancedb` para busca e base de dados.
"""

import lancedb

from estudai.domain.curriculum import Topico
from estudai.domain.material import TipoDocumento
from estudai.ports.knowledge import Document, SearchMode
from estudai.ports.llm import Embedder

DOCS_TABLE_NAME = "documents"


class KnowledgeDatabase:
    def __init__(self, db_uri: str, embedder: Embedder):
        "Loads a pre-built knowledge-base from a lancedb URI"
        self.db = lancedb.connect(db_uri)
        self.embedder = embedder

    @classmethod
    def build(cls, db_uri: str, documentos: list[Document], embedder: Embedder, **kw) -> "KnowledgeDatabase":
        "Builds a knowledge base from documents and an embedder. Returns a ready-to-use `KnowledgeDatabase` class"

        # Embed documents
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
    ) -> list[Document]:
        t = self.db.open_table(DOCS_TABLE_NAME)

        if mode == SearchMode.TEXTUAL:
            results = t.search(query, query_type="fts")
        else:
            vetor = self.embedder.embed([query], query=True)[0]
            if mode == SearchMode.SEMANTICA:
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

        return results.limit(k).to_pydantic(Document)

    def get(self, doc_id: str) -> Document | None:
        t = self.db.open_table(DOCS_TABLE_NAME)

        results = t.search().where(f"id = '{doc_id}'").limit(1).to_list()

        return results[0] if results else None
