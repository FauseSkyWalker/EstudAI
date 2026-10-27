"""Normalização e tokenização de texto em português (sem dependências externas)."""

import re
import unicodedata

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    [
        "a",
        "o",
        "as",
        "os",
        "um",
        "uma",
        "uns",
        "umas",
        "de",
        "do",
        "da",
        "dos",
        "das",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "por",
        "pelo",
        "pela",
        "pelos",
        "pelas",
        "para",
        "com",
        "sem",
        "sob",
        "sobre",
        "e",
        "ou",
        "que",
        "se",
        "ao",
        "aos",
        "à",
        "às",
        "é",
        "ser",
        "foi",
        "sao",
        "são",
        "como",
        "mais",
        "menos",
        "muito",
        "entre",
        "seu",
        "sua",
        "seus",
        "suas",
        "este",
        "esta",
        "esse",
        "essa",
        "isto",
        "isso",
        "aquele",
        "aquela",
        "ja",
        "já",
        "nao",
        "não",
        "tem",
        "ter",
    ]
)


def normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in sem_acento if not unicodedata.combining(c))


def tokenizar(texto: str) -> list[str]:
    return [t for t in _TOKEN.findall(normalizar(texto)) if t not in STOPWORDS]
