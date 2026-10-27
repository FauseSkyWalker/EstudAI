"""Agregação dos resultados: métricas da Seção 5.2, comparações da 5.3 e critérios da 5.1."""

from collections import defaultdict

import numpy as np
from pydantic import BaseModel

from estudai.evaluation.estatistica import bootstrap_ic, fleiss_kappa, wilcoxon_pareado
from estudai.evaluation.experimento import Condicao, ResultadoExecucao


class ResumoCondicao(BaseModel):
    condicao: Condicao
    execucoes: int
    falhas: int
    ganho_medio: float
    ganho_ic95: tuple[float, float]
    rubrica_media: float | None
    rubrica_ic95: tuple[float, float] | None
    rubrica_por_dimensao: dict[str, float]
    acuracia_diagnostico: float | None
    taxa_descarte: float | None
    validade_entregues: float | None
    itens_com_defeito_critico: float | None
    custo_medio_usd: float
    latencia_media_s: float


class Comparacao(BaseModel):
    metrica: str
    base: Condicao
    alvo: Condicao
    teste: dict[str, float]


class Relatorio(BaseModel):
    condicoes: list[ResumoCondicao]
    comparacoes: list[Comparacao]
    kappa_juizes_rubrica: float | None
    kappa_resolutores: float | None
    criterios: dict[str, bool | None]


_DIMENSOES = ("fidelidade_material", "personalizacao", "aplicabilidade", "profundidade_logica")


def _media(valores: list[float]) -> float | None:
    return float(np.mean(valores)) if valores else None


def _rubrica_execucao(r: ResultadoExecucao) -> float | None:
    notas = [n.media for c in r.ciclos for n in c.notas_tutoria]
    return _media(notas)


def resumir_condicao(condicao: Condicao, execucoes: list[ResultadoExecucao]) -> ResumoCondicao:
    ciclos = [c for r in execucoes for c in r.ciclos]
    ganhos = [r.ganho for r in execucoes]
    rubricas = [v for r in execucoes if (v := _rubrica_execucao(r)) is not None]
    notas = [n for c in ciclos for n in c.notas_tutoria]
    diagnosticados = [c for c in ciclos if c.nivel_estimado is not None]
    avaliacoes = [a for c in ciclos for item in c.avaliacoes_itens for a in item]
    entregues = sum(c.itens_entregues for c in ciclos)
    removidos = sum(c.itens_removidos for c in ciclos)
    validados = sum(c.itens_entregues + c.itens_removidos + c.itens_descartados for c in ciclos)
    gerou = condicao.fonte == "gerado"
    return ResumoCondicao(
        condicao=condicao,
        execucoes=len(execucoes),
        falhas=sum(r.erro is not None for r in execucoes),
        ganho_medio=float(np.mean(ganhos)) if ganhos else 0.0,
        ganho_ic95=bootstrap_ic(ganhos),
        rubrica_media=_media(rubricas),
        rubrica_ic95=bootstrap_ic(rubricas) if rubricas else None,
        rubrica_por_dimensao={d: float(np.mean([getattr(n, d) for n in notas])) for d in _DIMENSOES} if notas else {},
        acuracia_diagnostico=(
            _media([float(c.nivel_estimado == c.nivel_verdadeiro) for c in diagnosticados]) if diagnosticados else None
        ),
        taxa_descarte=(sum(c.itens_descartados for c in ciclos) / validados) if gerou and validados else None,
        validade_entregues=(1 - removidos / (entregues + removidos)) if gerou and entregues + removidos else None,
        itens_com_defeito_critico=_media([float(a.defeito_critico) for a in avaliacoes]),
        custo_medio_usd=float(np.mean([c.custo_usd for c in ciclos])) if ciclos else 0.0,
        latencia_media_s=float(np.mean([c.latencia_s for c in ciclos])) if ciclos else 0.0,
    )


def _pareados(resultados: list[ResultadoExecucao], metrica, base: Condicao, alvo: Condicao) -> dict[str, float]:
    por_chave: dict[tuple, dict[Condicao, float]] = defaultdict(dict)
    for r in resultados:
        if (valor := metrica(r)) is not None:
            por_chave[(r.persona_id, r.topico, r.seed)][r.condicao] = valor
    pares = [(v[base], v[alvo]) for v in por_chave.values() if base in v and alvo in v]
    return wilcoxon_pareado([p[0] for p in pares], [p[1] for p in pares])


def gerar_relatorio(
    resultados: list[ResultadoExecucao],
    *,
    rubrica_minima: float = 3.0,
    acuracia_minima: float = 0.8,
    validade_minima: float = 0.95,
    descarte_maximo: float = 0.30,
    orcamento_usd: float | None = None,
) -> Relatorio:
    por_condicao: dict[Condicao, list[ResultadoExecucao]] = defaultdict(list)
    for r in resultados:
        por_condicao[r.condicao].append(r)
    resumos = {c: resumir_condicao(c, rs) for c, rs in sorted(por_condicao.items())}

    comparacoes = [
        Comparacao(metrica=nome, base=Condicao.C1, alvo=alvo, teste=_pareados(resultados, fn, Condicao.C1, alvo))
        for alvo in (Condicao.C2, Condicao.C3)
        if alvo in resumos and Condicao.C1 in resumos
        for nome, fn in (("ganho", lambda r: r.ganho), ("rubrica", _rubrica_execucao))
    ]

    ciclos = [c for r in resultados for c in r.ciclos]
    juizes = [[round(n.media) for n in c.notas_tutoria] for c in ciclos if len(c.notas_tutoria) >= 2]
    votos = [v for c in ciclos for v in c.votos_painel if len(v) >= 2]
    kappa_juizes = fleiss_kappa(juizes) if juizes else None
    kappa_resolutores = fleiss_kappa(votos) if votos else None

    c1, c2 = resumos.get(Condicao.C1), resumos.get(Condicao.C2)
    criterios: dict[str, bool | None] = {
        "1_exercicios_confiaveis": (
            c1.validade_entregues >= validade_minima and c1.taxa_descarte <= descarte_maximo
            if c1 and c1.validade_entregues is not None and c1.taxa_descarte is not None
            else None
        ),
        "2_diagnostico_correto": (
            c1.acuracia_diagnostico >= acuracia_minima if c1 and c1.acuracia_diagnostico is not None else None
        ),
        "3_adaptatividade_util": (
            c1.ganho_medio > c2.ganho_medio and (c1.rubrica_media or 0) > (c2.rubrica_media or 0) if c1 and c2 else None
        ),
        "4_qualidade_tutoria": (
            all(v >= rubrica_minima for v in c1.rubrica_por_dimensao.values())
            if c1 and c1.rubrica_por_dimensao
            else None
        ),
        "5_eficiencia": (c1.custo_medio_usd <= orcamento_usd if c1 and orcamento_usd is not None else None),
    }
    return Relatorio(
        condicoes=list(resumos.values()),
        comparacoes=comparacoes,
        kappa_juizes_rubrica=kappa_juizes,
        kappa_resolutores=kappa_resolutores,
        criterios=criterios,
    )
