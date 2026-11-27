"""Exportação do relatório experimental em JSON, CSV e Markdown."""

import csv
from pathlib import Path

from estudai.evaluation.registros import ResultadoExecucao
from estudai.evaluation.relatorio import Comparacao, Relatorio

COLUNAS_RESUMO = (
    "condicao",
    "execucoes",
    "falhas",
    "ganho_medio",
    "rubrica_media",
    "sessoes_rubrica_ok",
    "acuracia_diagnostico_inicial",
    "acuracia_diagnostico_final",
    "taxa_descarte",
    "taxa_aprovacao_filtro",
    "validacao_auditada",
    "custo_medio_sessao_usd",
    "latencia_media_sessao_s",
)
COLUNAS_TOPICO = ("condicao", "topico", "execucoes", "ganho_medio", "rubrica_media", "acuracia_diagnostico_final")


def descricao_comparacao(comp: Comparacao) -> str:
    """Texto único (terminal e Markdown) com p, p ajustado por Holm e tamanho do efeito."""
    ajustado = comp.teste.get("p_ajustado")
    sufixo = f", p_Holm={ajustado:.4f}" if ajustado is not None else ""
    return (
        f"p={comp.teste.get('p_valor', float('nan')):.4f}{sufixo} "
        f"(n={comp.teste['n']:.0f}, mediana Δ={comp.teste['mediana_diferenca']:+.3f})"
    )


def exportar(rel: Relatorio, resultados: list[ResultadoExecucao], prefixo: Path) -> None:
    """Grava `.json` (relatório completo), `.csv` (execuções) e `.md` (relatório legível)."""
    prefixo.parent.mkdir(parents=True, exist_ok=True)
    prefixo.with_suffix(".json").write_text(rel.model_dump_json(indent=2), encoding="utf-8")
    _exportar_csv(resultados, prefixo.with_suffix(".csv"))
    prefixo.with_suffix(".md").write_text(para_markdown(rel), encoding="utf-8")


def para_markdown(rel: Relatorio) -> str:
    linhas = ["# Relatório experimental", "", "## Métricas por condição", ""]
    linhas += _tabela_md(COLUNAS_RESUMO, rel.condicoes)

    linhas += ["", "## Comparações", ""]
    for comp in rel.comparacoes:
        linhas.append(
            f"- **{comp.metrica}** {comp.base.value} vs {comp.alvo.value}: {descricao_comparacao(comp)}"
        )
    linhas.append(f"- ajuste de Holm sobre {rel.comparacoes_ajustadas} comparações planejadas")

    linhas += ["", "## Concordância", ""]
    linhas.append(f"- κ juízes (rubrica): {_valor(rel.kappa_juizes_rubrica)}")
    linhas.append(f"- concordância adjacente: {_valor(rel.concordancia_juizes_rubrica)}")
    linhas.append(f"- κ juízes (defeito crítico): {_valor(rel.kappa_juizes_itens)}")
    linhas.append(f"- κ resolutores: {_valor(rel.kappa_resolutores)}")

    linhas += ["", "## Critérios", ""]
    for nome, ok in rel.criterios.items():
        linhas.append(f"- {nome}: **{_valor(ok)}**")

    if rel.por_topico:
        linhas += ["", "## Métricas por tópico", ""]
        linhas += _tabela_md(COLUNAS_TOPICO, rel.por_topico)

    return "\n".join(linhas) + "\n"


def _exportar_csv(resultados: list[ResultadoExecucao], caminho: Path) -> None:
    campos = ["condicao", "persona_id", "topico", "seed", "pre_teste", "pos_teste", "ganho", "ciclos", "erro"]
    with caminho.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        for r in resultados:
            escritor.writerow(
                {
                    "condicao": r.condicao.value,
                    "persona_id": r.persona_id,
                    "topico": r.topico.value,
                    "seed": r.seed,
                    "pre_teste": f"{r.pre_teste:.6f}",
                    "pos_teste": f"{r.pos_teste:.6f}",
                    "ganho": f"{r.ganho:.6f}",
                    "ciclos": len(r.ciclos),
                    "erro": r.erro or "",
                }
            )


def _tabela_md(colunas: tuple[str, ...], resumos: list) -> list[str]:
    linhas = ["| " + " | ".join(colunas) + " |", "|" + "|".join("---" for _ in colunas) + "|"]
    linhas += ["| " + " | ".join(_valor(getattr(resumo, c)) for c in colunas) + " |" for resumo in resumos]
    return linhas


def _valor(valor) -> str:
    if valor is None:
        return "—"
    if isinstance(valor, bool):
        return "sim" if valor else "não"
    if isinstance(valor, float):
        return f"{valor:.3f}"
    if hasattr(valor, "value"):
        return str(valor.value)
    return str(valor)
