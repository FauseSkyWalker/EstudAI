"""Leitores das bases públicas de questões, normalizados para `QuestaoBruta`.

- BLUEX (Comvest/Fuvest): possui rótulo de disciplina e legendas de imagens (BLUEX Revisited).
- ENEM Challenge 2009-2017: não possui rótulo de disciplina (classificação via LLM).
- ENEM 2022-2024 (maritaca-ai/enem): Sem rótulo direto de disciplina,
imagens com `[[placeholder]]` + descrições textuais
"""

import re
from collections.abc import Iterator

from pydantic import BaseModel, Field

from estudai.domain.curriculum import Disciplina
from estudai.infrastructure.ingestion.normalizacao import limpar_texto, referencia_figura, texto_invalido

PLACEHOLDER = "[[placeholder]]"
_PREFIXO_ALTERNATIVA = re.compile(r"^\s*\(?[a-eA-E]\)\s*")
_MARCADOR_IMAGEM = re.compile(r"\[IMAGE (\d+)\]")


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
    texto_invalido: bool = False  # sobrou marcação/placeholder ou o MathML não converteu


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

        contexto = list(row.get("context_captions") or [])
        cegas = list(row.get("blind_captions") or [])
        enunciado, legendas = _substituir_imagens_bluex(row["question"], contexto, cegas)
        enunciado = limpar_texto(enunciado)
        alternativas = [limpar_texto(_PREFIXO_ALTERNATIVA.sub("", a).strip()) for a in row["alternatives"]]
        yield QuestaoBruta(
            id=f"BLUEX#{row['id']}",
            fonte="bluex",
            ano=_ano(row["id"]),
            enunciado=enunciado,
            alternativas=alternativas,
            gabarito=row["answer"],
            disciplina_indicada=indicada,
            descricoes_imagens=legendas,
            # sobrou marcador sem legenda, ou há imagem sem nenhum marcador/legenda
            depende_imagem_sem_descricao=bool(_MARCADOR_IMAGEM.search(enunciado))
            or (bool(row["has_associated_images"]) and not legendas),
            texto_invalido=texto_invalido(enunciado) or any(texto_invalido(a) for a in alternativas),
        )


def _substituir_imagens_bluex(enunciado: str, contexto: list[str], cegas: list[str]) -> tuple[str, list[str]]:
    """Troca cada `[IMAGE i]` pela legenda da imagem i. A legenda de contexto (gerada vendo o
    enunciado) é preferida à "cega"; marcadores sem legenda permanecem, e a questão é descartada."""
    usadas: list[str] = []

    def legenda(match: re.Match) -> str:
        i = int(match.group(1))
        texto = (contexto[i] if i < len(contexto) else "") or (cegas[i] if i < len(cegas) else "")
        if not texto:
            return match.group(0)
        usadas.append(texto)
        return f"[Figura: {texto.strip()}]"

    return _MARCADOR_IMAGEM.sub(legenda, enunciado), usadas or contexto or cegas


def carregar_enem_challenge() -> Iterator[QuestaoBruta]:
    for row in _load("eduagarcia/enem_challenge", split="train"):
        if row["nullified"] or int(row["exam_year"]) >= 2022:
            continue

        enunciado = limpar_texto(row["question"])
        alternativas = [limpar_texto(a) for a in row["choices"]["text"]]
        yield QuestaoBruta(
            id=f"ENEMC#{row['id']}",
            fonte="enem_challenge",
            ano=_ano(str(row["exam_year"])),
            enunciado=enunciado,
            alternativas=alternativas,
            gabarito=row["answerKey"],
            # a base não tem imagens: perguntas que dependem de figura são insolúveis
            depende_imagem_sem_descricao=referencia_figura(enunciado),
            texto_invalido=texto_invalido(enunciado) or any(texto_invalido(a) for a in alternativas),
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
            enunciado = limpar_texto(enunciado)
            alternativas = [limpar_texto(a) for a in alternativas]

            yield QuestaoBruta(
                id=f"ENEM{ano}#{row['id']}",
                fonte="enem_2022_2024",
                ano=int(ano),
                enunciado=enunciado,
                alternativas=alternativas,
                gabarito=row["label"],
                descricoes_imagens=descricoes,
                depende_imagem_sem_descricao=PLACEHOLDER in enunciado,
                texto_invalido=texto_invalido(enunciado) or any(texto_invalido(a) for a in alternativas),
            )


FONTES = {
    "bluex": carregar_bluex,
    "enem_challenge": carregar_enem_challenge,
    "enem_2022_2024": carregar_enem_2022_2024,
}
