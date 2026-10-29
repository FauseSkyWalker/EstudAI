"""Leitores das bases públicas de questões (Tabela 1), normalizados para `QuestaoBruta`.

- BLUEX (Comvest/Fuvest): tem rótulo de disciplina e legendas de imagens (BLUEX Revisited).
- ENEM Challenge 2009–2017: sem rótulo de disciplina (classificação via LLM).
- ENEM 2022–2024 (maritaca-ai/enem): figuras como `[[placeholder]]` + descrições textuais.
"""

import re
from collections.abc import Iterator

from pydantic import BaseModel, Field

from estudai.domain.curriculum import Disciplina

PLACEHOLDER = "[[placeholder]]"
_PREFIXO_ALTERNATIVA = re.compile(r"^\s*\(?[a-eA-E]\)\s*")


class QuestaoBruta(BaseModel):
    id: str
    fonte: str
    ano: int | None
    enunciado: str
    alternativas: list[str]
    gabarito: str
    disciplina_indicada: Disciplina | None = None  # metadado da base, quando existe
    descricoes_imagens: list[str] = Field(default_factory=list)
    depende_imagem_sem_descricao: bool = False


def _load(nome: str, **kw):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("instale o extra: uv sync --extra ingest") from exc
    return load_dataset(nome, **kw)


def _ano(texto: str) -> int | None:
    match = re.search(r"(19|20)\d{2}", texto)
    return int(match.group()) if match else None


def carregar_bluex() -> Iterator[QuestaoBruta]:
    disciplinas = {"mathematics": Disciplina.MATEMATICA, "physics": Disciplina.FISICA}
    for row in _load("portuguese-benchmark-datasets/BLUEX", split="questions"):
        indicada = next((disciplinas[s] for s in row["subject"] if s in disciplinas), None)
        if indicada is None or row["alternatives_type"] != "string" or not row["answer"]:
            continue

        legendas = list(row.get("blind_captions") or []) + list(row.get("context_captions") or [])

        yield QuestaoBruta(
            id=f"BLUEX#{row['id']}",
            fonte="bluex",
            ano=_ano(row["id"]),
            enunciado=row["question"],
            alternativas=[_PREFIXO_ALTERNATIVA.sub("", a).strip() for a in row["alternatives"]],
            gabarito=row["answer"],
            disciplina_indicada=indicada,
            descricoes_imagens=legendas,
            depende_imagem_sem_descricao=bool(row["has_associated_images"]) and not legendas,
        )


def carregar_enem_challenge() -> Iterator[QuestaoBruta]:
    for row in _load("eduagarcia/enem_challenge", split="train"):
        if row["nullified"] or int(row["exam_year"]) >= 2022:
            continue

        yield QuestaoBruta(
            id=f"ENEMC#{row['id']}",
            fonte="enem_challenge",
            ano=_ano(str(row["exam_year"])),
            enunciado=row["question"],
            alternativas=list(row["choices"]["text"]),
            gabarito=row["answerKey"],
        )


def carregar_enem_2022_2024() -> Iterator[QuestaoBruta]:
    for ano in ("2022", "2023", "2024"):
        for row in _load("maritaca-ai/enem", name=ano, split="train"):
            alternativas = list(row["alternatives"])
            if any(PLACEHOLDER in a for a in alternativas):
                continue  # alternativas em imagem

            descricoes = list(row.get("description") or [])
            enunciado = row["question"]
            for desc in descricoes:
                enunciado = enunciado.replace(PLACEHOLDER, f"[Figura: {desc}]", 1)

            yield QuestaoBruta(
                id=f"ENEM{ano}#{row['id']}",
                fonte="enem_2022_2024",
                ano=int(ano),
                enunciado=enunciado,
                alternativas=alternativas,
                gabarito=row["label"],
                descricoes_imagens=descricoes,
                depende_imagem_sem_descricao=PLACEHOLDER in enunciado,
            )


FONTES = {
    "bluex": carregar_bluex,
    "enem_challenge": carregar_enem_challenge,
    "enem_2022_2024": carregar_enem_2022_2024,
}
