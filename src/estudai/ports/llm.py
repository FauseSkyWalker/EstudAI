"""Contrato agnóstico de fornecedor para modelos de linguagem e embeddings."""

from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class ModelTier(StrEnum):
    """Níveis de capacidade da Seção 3, em vez de fornecedores fixos."""

    FRONTEIRA = "fronteira"
    ECONOMICO = "economico"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    system: str
    messages: list[ChatMessage]
    json_schema: dict[str, Any] | None = None  # pede saída JSON aderente a este schema
    temperature: float = 0.2
    max_output_tokens: int | None = None
    caller: str = "desconhecido"  # identifica o agente (telemetria de custo por agente)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    text: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0.0


class LLMError(Exception):
    """Falha definitiva na chamada ao modelo."""


class TransientLLMError(LLMError):
    """Falha recuperável (rate limit, timeout, 5xx): elegível para retry."""


class LLMClient(Protocol):
    @property
    def model(self) -> str: ...

    def complete(self, request: LLMRequest) -> LLMResponse: ...


class Embedder(Protocol):
    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        """`query=True` para consultas, False para documentos (alguns modelos distinguem)."""
        ...
