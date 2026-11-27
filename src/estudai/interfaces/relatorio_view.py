"""Renderização do relatório experimental no terminal."""

from rich.console import Console
from rich.table import Table

from estudai.evaluation.exportacao import COLUNAS_RESUMO, COLUNAS_TOPICO, descricao_comparacao
from estudai.evaluation.relatorio import Relatorio


def imprimir_relatorio(rel: Relatorio, console: Console) -> None:
    console.print(_tabela("Métricas por condição", COLUNAS_RESUMO, rel.condicoes))

    for resumo in rel.condicoes:
        if resumo.qualidade_itens:
            detalhes = " · ".join(f"{k}={v:.2f}" for k, v in resumo.qualidade_itens.items())
            console.print(f"[dim]{resumo.condicao} qualidade dos itens: {detalhes}[/dim]")

    if rel.por_topico:
        console.print(_tabela("Métricas por tópico", COLUNAS_TOPICO, rel.por_topico))

    for comp in rel.comparacoes:
        console.print(f"Wilcoxon {comp.metrica}: {comp.base} vs {comp.alvo} → {descricao_comparacao(comp)}")

    console.print(
        f"κ juízes — rubrica: {_fmt(rel.kappa_juizes_rubrica)} "
        f"(diferença ≤ 1 ponto em {_fmt(rel.concordancia_juizes_rubrica)}) · "
        f"transições: {_fmt(rel.kappa_transicoes)} · "
        f"defeito crítico nos exercícios: {_fmt(rel.kappa_juizes_itens)} · "
        f"κ resolutores: {_fmt(rel.kappa_resolutores)}"
    )
    if rel.kappa_juizes_rubrica is None:
        console.print("[dim]κ entre juízes exige 2+ juízes em ESTUDAI_JUDGES.[/dim]")
    for nome, ok in rel.criterios.items():
        marca = {True: "[green]atingido[/green]", False: "[red]não atingido[/red]", None: "[dim]n/a[/dim]"}[ok]
        console.print(f"Critério {nome}: {marca}")


def _tabela(titulo: str, colunas: tuple[str, ...], resumos: list) -> Table:
    tabela = Table(title=titulo)
    for coluna in colunas:
        tabela.add_column(coluna)
    for resumo in resumos:
        tabela.add_row(*[_fmt(getattr(resumo, c)) for c in colunas])
    return tabela


def _fmt(valor) -> str:
    if isinstance(valor, float):
        return f"{valor:.3f}"
    return "-" if valor is None else str(valor)
