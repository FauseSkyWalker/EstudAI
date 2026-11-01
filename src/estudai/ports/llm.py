"""Contrato de embeddings.

Para modelos de chat, a porta é o `BaseChatModel` do LangChain (decisão da migração para
LangChain/LangGraph); os adaptadores concretos ficam em `infrastructure/llm/factory.py`.
"""

from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        """`query=True` para consultas, False para documentos (alguns modelos distinguem)."""
        ...
