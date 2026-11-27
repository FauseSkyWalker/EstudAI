"""Agregação dos resultados: métricas da Seção 5.2, comparações da 5.3 e critérios da 5.1."""

from collections import defaultdict

import numpy as np
from pydantic import BaseModel, Field, computed_field

from estudai.domain.curriculum import Topico
from estudai.evaluation.estatistica import bootstrap_ic, fleiss_kappa, holm, media, wilcoxon_pareado
from estudai.evaluation.registros import Condicao, ResultadoExecucao


class ResumoCondicao(BaseModel):
    condicao: Condicao
    execucoes: int
    execucoes_validas: int
    falhas: int
    ganho_medio: float
    ganho_ic95: tuple[float, float]
    rubrica_media: float | None
    rubrica_ic95: tuple[float, float] | None
    rubrica_por_dimensao: dict[str, float]
    sessoes_rubrica_ok: float | None
    acuracia_diagnostico_inicial: float | None
    acuracia_diagnostico_final: float | None
    taxa_descarte: float | None
    validade_entregues: float | None
    itens_com_defeito_critico: float | None
    qualidade_itens: dict[str, float]
    custo_medio_usd: float
    custo_medio_sessao_usd: float
    latencia_media_s: float
    latencia_media_sessao_s: float
    tokens_entrada_medio: float
    tokens_saida_medio: float

    @computed_field
    @property
    def taxa_aprovacao_filtro(self) -> float | None:
        """Fração dos itens gerados que passou pelo filtro de validação."""
        return None if self.taxa_descarte is None else 1 - self.taxa_descarte

    @computed_field
    @property
    def validacao_auditada(self) -> float | None:
        """Fração dos itens entregues sem defeito crítico segundo os juízes."""
        return None if self.itens_com_defeito_critico is None else 1 - self.itens_com_defeito_critico


class ResumoTopico(BaseModel):
    condicao: Condicao
    topico: Topico
    execucoes: int
    ganho_medio: float
    rubrica_media: float | None
    acuracia_diagnostico_final: float | None


class Comparacao(BaseModel):
    metrica: str
    base: Condicao
    alvo: Condicao
    teste: dict[str, float]


class Relatorio(BaseModel):
    condicoes: list[ResumoCondicao]
    por_topico: list[ResumoTopico] = Field(default_factory=list)
    comparacoes: list[Comparacao]
    comparacoes_ajustadas: int = 0
    kappa_juizes_rubrica: float | None  # Fleiss sobre (sessão × dimensão), notas 1–5
    concordancia_juizes_rubrica: float | None  # fração de notas em que os juízes diferem em até 1 ponto
    kappa_juizes_itens: float | None  # Fleiss sobre `defeito_critico` de cada exercício
    kappa_resolutores: float | None
    criterios: dict[str, bool | None]


_DIMENSOES = ("fidelidade_material", "personalizacao", "aplicabilidade", "profundidade_logica")
_QUALIDADE_ITENS = (
    "vinculo_topico",
    "clareza",
    "plausibilidade_distratores",
    "sem_pistas_gramaticais",
    "defeito_critico",
)


def _completos(avaliacoes: list[list]) -> list[list]:
    """Fleiss exige o mesmo número de avaliadores por item: descarta itens em que algum juiz falhou."""
    n = max((len(a) for a in avaliacoes), default=0)
    return [a for a in avaliacoes if n >= 2 and len(a) == n]


def _kappa(avaliacoes: list[list]) -> float | None:
    completos = _completos(avaliacoes)
    return fleiss_kappa(completos) if completos else None


def _concordancia_adjacente(avaliacoes: list[list[int]]) -> float | None:
    completos = _completos(avaliacoes)
    return media([float(max(a) - min(a) <= 1) for a in completos]) if completos else None


def _validas(execucoes: list[ResultadoExecucao]) -> list[ResultadoExecucao]:
    return [r for r in execucoes if r.erro is None]


def _rubrica_execucao(r: ResultadoExecucao) -> float | None:
    notas = [n.media for c in r.ciclos for n in c.notas_tutoria]
    return media(notas)


def _acuracia(ciclos, *, inicial: bool) -> float | None:
    if inicial:
        pares = [
            (c.nivel_estimado_inicial, c.nivel_verdadeiro_inicial)
            for c in ciclos
            if c.nivel_estimado_inicial is not None
        ]
    else:
        pares = [
            (c.nivel_estimado_final, c.nivel_verdadeiro_final) for c in ciclos if c.nivel_estimado_final is not None
        ]
    return media([float(estimado == verdadeiro) for estimado, verdadeiro in pares]) if pares else None


def _sessoes_rubrica_ok(ciclos, minimo: float) -> float | None:
    """Fração de ciclos em que TODAS as quatro dimensões atingem o mínimo, na média dos juízes."""
    avaliados = []
    for ciclo in ciclos:
        if not ciclo.notas_tutoria:
            continue
        medias = {
            dimensao: float(np.mean([getattr(nota, dimensao) for nota in ciclo.notas_tutoria]))
            for dimensao in _DIMENSOES
        }
        avaliados.append(all(valor >= minimo for valor in medias.values()))
    return media([float(ok) for ok in avaliados]) if avaliados else None


def _qualidade_itens(avaliacoes) -> dict[str, float]:
    if not avaliacoes:
        return {}
    return {
        dimensao: float(np.mean([float(getattr(avaliacao, dimensao)) for avaliacao in avaliacoes]))
        for dimensao in _QUALIDADE_ITENS
    }


def _contagens_itens(ciclos) -> tuple[int, int, int]:
    """(entregues, removidos, descartados) somados sobre os ciclos."""
    return (
        sum(c.itens_entregues for c in ciclos),
        sum(c.itens_removidos for c in ciclos),
        sum(c.itens_descartados for c in ciclos),
    )


def resumir_condicao(
    condicao: Condicao,
    execucoes: list[ResultadoExecucao],
    *,
    rubrica_minima: float = 3.0,
) -> ResumoCondicao:
    validas = _validas(execucoes)
    ciclos = [c for r in validas for c in r.ciclos]
    ganhos = [r.ganho for r in validas]
    rubricas = [v for r in validas if (v := _rubrica_execucao(r)) is not None]
    notas = [n for c in ciclos for n in c.notas_tutoria]
    avaliacoes = [a for c in ciclos for item in c.avaliacoes_itens for a in item]
    qualidade = _qualidade_itens(avaliacoes)
    defeito = qualidade.get("defeito_critico")
    gerou = condicao.fonte == "gerado"
    custos_sessao = [sum(c.custo_usd for c in r.ciclos) for r in validas]
    latencias_sessao = [sum(c.latencia_s for c in r.ciclos) for r in validas]
    acuracia_inicial = _acuracia(ciclos, inicial=True)
    acuracia_final = _acuracia(ciclos, inicial=False)
    entregues, removidos, descartados = _contagens_itens(ciclos)
    return ResumoCondicao(
        condicao=condicao,
        execucoes=len(execucoes),
        execucoes_validas=len(validas),
        falhas=sum(r.erro is not None for r in execucoes),
        ganho_medio=float(np.mean(ganhos)) if ganhos else 0.0,
        ganho_ic95=bootstrap_ic(ganhos),
        rubrica_media=media(rubricas),
        rubrica_ic95=bootstrap_ic(rubricas) if rubricas else None,
        rubrica_por_dimensao={d: float(np.mean([getattr(n, d) for n in notas])) for d in _DIMENSOES} if notas else {},
        sessoes_rubrica_ok=_sessoes_rubrica_ok(ciclos, rubrica_minima),
        acuracia_diagnostico_inicial=acuracia_inicial,
        acuracia_diagnostico_final=acuracia_final,
        taxa_descarte=(descartados / (entregues + descartados)) if gerou and entregues + descartados else None,
        validade_entregues=(1 - removidos / (entregues + removidos)) if gerou and entregues + removidos else None,
        itens_com_defeito_critico=defeito,
        qualidade_itens={d: v for d, v in qualidade.items() if d != "defeito_critico"},
        custo_medio_usd=float(np.mean([c.custo_usd for c in ciclos])) if ciclos else 0.0,
        custo_medio_sessao_usd=float(np.mean(custos_sessao)) if custos_sessao else 0.0,
        latencia_media_s=float(np.mean([c.latencia_s for c in ciclos])) if ciclos else 0.0,
        latencia_media_sessao_s=float(np.mean(latencias_sessao)) if latencias_sessao else 0.0,
        tokens_entrada_medio=float(np.mean([c.tokens_entrada for c in ciclos])) if ciclos else 0.0,
        tokens_saida_medio=float(np.mean([c.tokens_saida for c in ciclos])) if ciclos else 0.0,
    )


def resumir_topico(condicao: Condicao, topico: Topico, execucoes: list[ResultadoExecucao]) -> ResumoTopico:
    validas = _validas(execucoes)
    ciclos = [c for r in validas for c in r.ciclos]
    return ResumoTopico(
        condicao=condicao,
        topico=topico,
        execucoes=len(execucoes),
        ganho_medio=float(np.mean([r.ganho for r in validas])) if validas else 0.0,
        rubrica_media=media([v for r in validas if (v := _rubrica_execucao(r)) is not None]),
        acuracia_diagnostico_final=_acuracia(ciclos, inicial=False),
    )


def _pareados(resultados: list[ResultadoExecucao], metrica, base: Condicao, alvo: Condicao) -> dict[str, float]:
    por_chave: dict[tuple, dict[Condicao, float]] = defaultdict(dict)
    for r in _validas(resultados):
        if (valor := metrica(r)) is not None:
            por_chave[(r.persona_id, r.topico, r.seed)][r.condicao] = valor
    pares = [(v[base], v[alvo]) for v in por_chave.values() if base in v and alvo in v]
    return wilcoxon_pareado([p[0] for p in pares], [p[1] for p in pares])


def _ajustar_holm(comparacoes: list[Comparacao]) -> None:
    """Corrige os p-valores da família de comparações planejadas (Holm)."""
    pvals = [c.teste.get("p_valor", 1.0) for c in comparacoes]
    for comparacao, ajustado in zip(comparacoes, holm(pvals), strict=True):
        comparacao.teste["p_ajustado"] = ajustado


def gerar_relatorio(
    resultados: list[ResultadoExecucao],
    *,
    rubrica_minima: float = 3.0,
    sessoes_minimas: float = 0.8,
    acuracia_minima: float = 0.8,
    validade_minima: float = 0.95,
    descarte_maximo: float = 0.30,
    orcamento_usd: float | None = None,
) -> Relatorio:
    por_condicao: dict[Condicao, list[ResultadoExecucao]] = defaultdict(list)
    por_topico: dict[tuple[Condicao, Topico], list[ResultadoExecucao]] = defaultdict(list)
    for r in resultados:
        por_condicao[r.condicao].append(r)
        por_topico[(r.condicao, r.topico)].append(r)
    resumos = {c: resumir_condicao(c, rs, rubrica_minima=rubrica_minima) for c, rs in sorted(por_condicao.items())}
    topicos = [
        resumir_topico(c, t, rs)
        for (c, t), rs in sorted(por_topico.items(), key=lambda item: (item[0][0], item[0][1].value))
    ]

    comparacoes = [
        Comparacao(metrica=nome, base=Condicao.C1, alvo=alvo, teste=_pareados(resultados, fn, Condicao.C1, alvo))
        for alvo in (Condicao.C2, Condicao.C3)
        if alvo in resumos and Condicao.C1 in resumos
        for nome, fn in (("ganho", lambda r: r.ganho), ("rubrica", _rubrica_execucao))
    ]
    _ajustar_holm(comparacoes)

    ciclos = [c for r in _validas(resultados) for c in r.ciclos]
    # concordância entre juízes (Seção 5.3): cada (sessão, dimensão) e cada exercício é um item
    notas_rubrica = [[getattr(n, d) for n in c.notas_tutoria] for c in ciclos for d in _DIMENSOES]
    defeitos = [[a.defeito_critico for a in item] for c in ciclos for item in c.avaliacoes_itens]
    votos = [v for c in ciclos for v in c.votos_painel]

    c1, c2 = resumos.get(Condicao.C1), resumos.get(Condicao.C2)
    acuracia_c1 = (
        c1.acuracia_diagnostico_final
        if c1 and c1.acuracia_diagnostico_final is not None
        else (c1.acuracia_diagnostico_inicial if c1 else None)
    )
    criterios: dict[str, bool | None] = {
        "1_exercicios_confiaveis": (
            c1.validacao_auditada >= validade_minima and c1.taxa_descarte <= descarte_maximo
            if c1 and c1.validacao_auditada is not None and c1.taxa_descarte is not None
            else None
        ),
        "2_diagnostico_correto": (acuracia_c1 >= acuracia_minima if acuracia_c1 is not None else None),
        "3_adaptatividade_util": (
            c1.ganho_medio > c2.ganho_medio and (c1.rubrica_media or 0) > (c2.rubrica_media or 0) if c1 and c2 else None
        ),
        "4_qualidade_tutoria": (
            c1.sessoes_rubrica_ok >= sessoes_minimas if c1 and c1.sessoes_rubrica_ok is not None else None
        ),
        "5_eficiencia": (c1.custo_medio_sessao_usd <= orcamento_usd if c1 and orcamento_usd is not None else None),
    }
    return Relatorio(
        condicoes=list(resumos.values()),
        por_topico=topicos,
        comparacoes=comparacoes,
        comparacoes_ajustadas=len(comparacoes),
        kappa_juizes_rubrica=_kappa(notas_rubrica),
        concordancia_juizes_rubrica=_concordancia_adjacente(notas_rubrica),
        kappa_juizes_itens=_kappa(defeitos),
        kappa_resolutores=_kappa(votos),
        criterios=criterios,
    )
