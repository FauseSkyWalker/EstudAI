"""Coleta do material didático do pt.wikibooks.org (API MediaWiki, extensão TextExtracts)."""

import re
import tomllib
from collections.abc import Iterator
from pathlib import Path

import httpx
from pydantic import BaseModel

from estudai.domain.curriculum import Topico

API = "https://pt.wikibooks.org/w/api.php"
USER_AGENT = "EstudAI/0.1 (https://ic.unicamp.br; projeto academico MO810A) python-httpx"
_SECAO = re.compile(r"^(=+)\s*(.+?)\s*\1\s*$")


class PaginaWiki(BaseModel):
    titulo: str
    topico: Topico
    texto: str


class Trecho(BaseModel):
    id: str
    titulo: str
    secao: str
    topico: Topico
    texto: str


def ler_curadoria(caminho: Path) -> list[tuple[Topico, str]]:
    dados = tomllib.loads(caminho.read_text(encoding="utf-8"))
    return [(Topico(p["topico"]), t) for p in dados["pagina"] for t in p["titulos"]]


def baixar_paginas(
    curadoria: list[tuple[Topico, str]], client: httpx.Client | None = None
) -> Iterator[PaginaWiki | str]:
    """Produz `PaginaWiki` para cada título encontrado, ou o título (str) quando ausente."""
    client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30)
    for topico, titulo in curadoria:
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": titulo,
            "format": "json",
            "formatversion": 2,
        }
        pagina = client.get(API, params=params).raise_for_status().json()["query"]["pages"][0]
        texto = pagina.get("extract", "")
        if pagina.get("missing") or not texto.strip():
            yield titulo
        else:
            yield PaginaWiki(titulo=pagina["title"], topico=topico, texto=texto)


def fatiar(pagina: PaginaWiki, max_chars: int = 1200, sobreposicao: int = 200) -> list[Trecho]:
    """Divide por seção e, dentro de cada seção, em janelas de parágrafos com sobreposição."""
    secoes: list[tuple[str, list[str]]] = [("Introdução", [])]
    for linha in pagina.texto.splitlines():
        if match := _SECAO.match(linha.strip()):
            secoes.append((match.group(2), []))
        elif linha.strip():
            secoes[-1][1].append(linha.strip())

    trechos = []
    for secao, paragrafos in secoes:
        janela = ""
        for p in paragrafos:
            if janela and len(janela) + len(p) > max_chars:
                trechos.append((secao, janela))
                janela = janela[-sobreposicao:]
            janela = f"{janela}\n{p}".strip()
        if len(janela) > 80:  # ignora seções vazias/triviais
            trechos.append((secao, janela))

    slug = re.sub(r"\W+", "-", pagina.titulo.lower()).strip("-")
    return [
        Trecho(
            id=f"wb-{slug}-{i}",
            titulo=pagina.titulo,
            secao=secao,
            topico=pagina.topico,
            texto=f"{pagina.titulo} — {secao}\n{texto}",
        )
        for i, (secao, texto) in enumerate(trechos)
    ]
