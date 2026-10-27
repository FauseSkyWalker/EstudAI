"""Contratos de acesso aos dados de longo prazo: base vetorial (RAG) e banco de questões."""

from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from estudai.domain.curriculum import Topico
from estudai.domain.material import TipoDocumento
from estudai.domain.questoes import Particao, Questao


class SearchMode(StrEnum):
    SEMANTICA = "semantica"
    TEXTUAL = "textual"
    HIBRIDA = "hibrida"


class Document(BaseModel):
    id: str
    tipo: TipoDocumento
    fonte: str
    texto: str
    topico: Topico | None = None
    metadados: dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    document: Document
    score: float  # normalizado em [0, 1]


class KnowledgeBase(Protocol):
    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.HIBRIDA,
        k: int = 5,
        tipo: TipoDocumento | None = None,
        topico: Topico | None = None,
    ) -> list[SearchHit]: ...

    def get(self, doc_id: str) -> Document | None: ...


class QuestionBank(Protocol):
    def find(self, topico: Topico | None = None, particao: Particao | None = None) -> list[Questao]: ...

    def get(self, questao_id: str) -> Questao | None: ...
