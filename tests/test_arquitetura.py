"""Garante a regra de dependência da Clean Architecture: camadas internas não importam externas."""

import ast
from pathlib import Path

import pytest

RAIZ = Path(__file__).parents[1] / "src" / "estudai"

PERMITIDAS = {
    "domain": {"domain"},
    "ports": {"domain", "ports"},
    "agents": {"domain", "ports", "agents"},
    "application": {"domain", "ports", "agents", "application"},
    "infrastructure": {"domain", "ports", "agents", "application", "infrastructure"},
    "evaluation": {"domain", "ports", "agents", "application", "evaluation"},
}


def _imports_internos(arquivo: Path) -> set[str]:
    camadas = set()
    for no in ast.walk(ast.parse(arquivo.read_text(encoding="utf-8"))):
        nomes = [no.module or ""] if isinstance(no, ast.ImportFrom) else []
        nomes += [a.name for a in no.names] if isinstance(no, ast.Import) else []
        camadas |= {n.split(".")[1] for n in nomes if n.startswith("estudai.") and n.count(".") >= 1}
    return camadas


@pytest.mark.parametrize("camada", sorted(PERMITIDAS))
def test_regra_de_dependencia(camada):
    violacoes = {
        str(arquivo.relative_to(RAIZ)): proibidas
        for arquivo in (RAIZ / camada).rglob("*.py")
        if (proibidas := _imports_internos(arquivo) - PERMITIDAS[camada])
    }
    assert not violacoes, f"{camada} importa camadas externas: {violacoes}"


def test_sdks_de_fornecedor_so_na_infraestrutura():
    for arquivo in RAIZ.rglob("*.py"):
        if "infrastructure" in arquivo.parts:
            continue
        texto = arquivo.read_text(encoding="utf-8")
        assert "from google" not in texto and "import openai" not in texto, arquivo
