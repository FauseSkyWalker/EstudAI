"""Taxonomia curricular: disciplinas, tópicos avaliados (Seção 5.3) e seus conceitos.

O conceito é a unidade de granularidade compartilhada por todo o sistema: o perfil do aluno
registra lacunas por conceito, as questões são rotuladas por conceito e o estudante simulado
é definido por um mapa de conceitos dominados.
"""

from enum import StrEnum


class Disciplina(StrEnum):
    MATEMATICA = "matematica"
    FISICA = "fisica"


class Topico(StrEnum):
    FUNCOES = "funcoes"
    PROBABILIDADE = "probabilidade"
    CINEMATICA = "cinematica"
    DINAMICA = "dinamica"

    @property
    def disciplina(self) -> Disciplina:
        return DISCIPLINA_DO_TOPICO[self]

    @property
    def conceitos(self) -> tuple[str, ...]:
        return CONCEITOS[self]


DISCIPLINA_DO_TOPICO: dict[Topico, Disciplina] = {
    Topico.FUNCOES: Disciplina.MATEMATICA,
    Topico.PROBABILIDADE: Disciplina.MATEMATICA,
    Topico.CINEMATICA: Disciplina.FISICA,
    Topico.DINAMICA: Disciplina.FISICA,
}

CONCEITOS: dict[Topico, tuple[str, ...]] = {
    Topico.FUNCOES: (
        "dominio_imagem",
        "funcao_afim",
        "funcao_quadratica",
        "funcao_exponencial",
        "funcao_logaritmica",
        "composicao_inversa",
    ),
    Topico.PROBABILIDADE: (
        "principio_contagem",
        "permutacoes_combinacoes",
        "probabilidade_classica",
        "probabilidade_condicional",
        "eventos_independentes",
    ),
    Topico.CINEMATICA: (
        "movimento_uniforme",
        "movimento_uniformemente_variado",
        "queda_livre",
        "lancamento_obliquo",
        "movimento_circular",
    ),
    Topico.DINAMICA: (
        "leis_de_newton",
        "atrito",
        "plano_inclinado",
        "trabalho_energia",
        "quantidade_de_movimento",
    ),
}


def topico_do_conceito(conceito: str) -> Topico | None:
    for topico, conceitos in CONCEITOS.items():
        if conceito in conceitos:
            return topico
    return None


def conceitos_validos(topico: Topico, conceitos: list[str]) -> list[str]:
    """Filtra conceitos que não pertencem ao tópico (saídas de LLM podem inventar rótulos)."""
    permitidos = set(topico.conceitos)
    return [c for c in dict.fromkeys(conceitos) if c in permitidos]
