"""BM25 Okapi para a busca textual."""

import math
from collections import Counter

import numpy as np

from estudai.infrastructure.knowledge.text import tokenizar


class BM25Index:
    def __init__(self, textos: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1, self._b = k1, b
        self._docs = [Counter(tokenizar(t)) for t in textos]
        self._lens = np.array([sum(d.values()) for d in self._docs], dtype=float)
        self._avgdl = float(self._lens.mean()) if len(self._docs) else 0.0
        df = Counter(term for d in self._docs for term in d)
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, consulta: str) -> np.ndarray:
        resultado = np.zeros(len(self._docs))
        termos = [t for t in tokenizar(consulta) if t in self._idf]
        if not termos:
            return resultado
        norm = self._k1 * (1 - self._b + self._b * self._lens / (self._avgdl or 1.0))
        for i, doc in enumerate(self._docs):
            s = 0.0
            for t in termos:
                tf = doc.get(t, 0)
                if tf:
                    s += self._idf[t] * tf * (self._k1 + 1) / (tf + norm[i])
            resultado[i] = s
        return resultado
