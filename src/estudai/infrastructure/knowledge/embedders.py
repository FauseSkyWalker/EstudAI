"""Embedder offline por feature hashing (palavras + trigramas de caracteres).

Determinístico e sem custo: útil em testes e para rodar o pipeline sem chave de embeddings.
A qualidade semântica é limitada; nos experimentos use o embedder do fornecedor.
"""

import hashlib

import numpy as np

from estudai.infrastructure.knowledge.text import normalizar, tokenizar


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self._dim = dim

    def _indice(self, feature: str) -> int:
        return int.from_bytes(hashlib.md5(feature.encode()).digest()[:4], "little") % self._dim

    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        vetores = []
        for texto in texts:
            v = np.zeros(self._dim)
            for token in tokenizar(texto):
                v[self._indice(f"w:{token}")] += 1.0
            compacto = normalizar(texto).replace(" ", "_")
            for i in range(len(compacto) - 2):
                v[self._indice(f"c:{compacto[i : i + 3]}")] += 0.3
            norma = np.linalg.norm(v)
            vetores.append((v / norma if norma else v).tolist())
        return vetores
