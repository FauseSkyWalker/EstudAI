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


ROTULOS_CONCEITOS: dict[str, str] = {
    "dominio_imagem": "Domínio e imagem",
    "funcao_afim": "Função afim",
    "funcao_quadratica": "Função quadrática",
    "funcao_exponencial": "Função exponencial",
    "funcao_logaritmica": "Função logarítmica",
    "composicao_inversa": "Funções compostas e inversas",
    "principio_contagem": "Princípio da contagem",
    "permutacoes_combinacoes": "Permutações e combinações",
    "probabilidade_classica": "Probabilidade clássica",
    "probabilidade_condicional": "Probabilidade condicional",
    "eventos_independentes": "Eventos independentes",
    "movimento_uniforme": "Movimento uniforme",
    "movimento_uniformemente_variado": "Movimento uniformemente variado",
    "queda_livre": "Queda livre",
    "lancamento_obliquo": "Lançamento oblíquo",
    "movimento_circular": "Movimento circular",
    "leis_de_newton": "Leis de Newton",
    "atrito": "Atrito",
    "plano_inclinado": "Plano inclinado",
    "trabalho_energia": "Trabalho e energia",
    "quantidade_de_movimento": "Quantidade de movimento",
}


def rotulo_conceito(conceito: str) -> str:
    """Rótulo legível para exibição; IDs desconhecidos viram texto simples."""
    return ROTULOS_CONCEITOS.get(conceito, conceito.replace("_", " "))


def topico_do_conceito(conceito: str) -> Topico | None:
    for topico, conceitos in CONCEITOS.items():
        if conceito in conceitos:
            return topico
    return None


def conceitos_validos(topico: Topico, conceitos: list[str]) -> list[str]:
    """Filtra conceitos que não pertencem ao tópico (saídas de LLM podem inventar rótulos)."""
    permitidos = set(topico.conceitos)
    return [c for c in dict.fromkeys(conceitos) if c in permitidos]
