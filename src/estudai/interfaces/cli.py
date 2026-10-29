"""CLI do EstudAI.

Fluxo típico:
    estudai ingest-questions      # baixa, filtra e rotula o banco de questões
    estudai ingest-wikibooks      # baixa e fatia o material didático
    estudai build-index           # monta a base vetorial
    estudai estudar --aluno ana --topico funcoes
    estudai experimento --saida results/exp1.jsonl
    estudai relatorio results/exp1.jsonl
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from estudai.domain.curriculum import Topico

# Em consoles/redirecionamentos não-UTF-8 (cp1252 no Windows), substitui em vez de quebrar
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

app = typer.Typer(help="Pipeline multiagente de tutoria adaptativa em Matemática e Física.", no_args_is_help=True)
console = Console()


def _container():
    from estudai.bootstrap import Container
    from estudai.settings import Settings

    return Container(Settings())


@app.command("ingest-questions")
def ingest_questions(
    fontes: Annotated[list[str], typer.Option(help="bluex, enem_challenge, enem_2022_2024")] = [
        "bluex",
        "enem_challenge",
        "enem_2022_2024",
    ],
    bateria: Annotated[int, typer.Option(help="Questões por tópico na bateria de pré/pós-teste")] = 8,
    limite: Annotated[int | None, typer.Option(help="Máx. de questões por fonte (para testes)")] = None,
    workers: int = 8,
) -> None:
    """Baixa as bases públicas, rotula com o modelo econômico e grava o banco de questões."""
    from itertools import chain, islice

    from estudai.infrastructure.ingestion.classificador import ClassificadorQuestoes
    from estudai.infrastructure.ingestion.questoes import RotuloCache, ingerir_questoes
    from estudai.infrastructure.ingestion.sources import FONTES
    from estudai.infrastructure.knowledge.question_bank import JsonlQuestionBank

    container = _container()
    s = container.settings
    brutas = chain.from_iterable(islice(FONTES[f](), limite) for f in fontes)

    questoes, stats = ingerir_questoes(
        brutas,
        ClassificadorQuestoes(container.llm_economico, temperature=0.0),
        RotuloCache(s.rotulos_cache_path),
        bateria_por_topico=bateria,
        workers=workers,
        progresso=console.print,
    )
    JsonlQuestionBank.save(questoes, s.questoes_path)
    console.print_json(json.dumps(stats))
    console.print(f"[green]{len(questoes)} questões gravadas em {s.questoes_path}[/green]")
    _imprimir_uso(container)


@app.command("ingest-wikibooks")
def ingest_wikibooks(config: Path = Path("config/wikibooks.toml")) -> None:
    """Baixa as páginas do Wikibooks indicadas em `config/wikibooks.toml` e aplica chunking."""
    from estudai.infrastructure.ingestion.wikibooks import PaginaWiki, baixar_paginas, fatiar, ler_curadoria
    from estudai.settings import Settings

    destino = Settings().trechos_path
    destino.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with destino.open("w", encoding="utf-8") as f:
        for pagina in baixar_paginas(ler_curadoria(config)):
            if not isinstance(pagina, PaginaWiki):
                console.print(f"[yellow]página não encontrada: {pagina}[/yellow]")
                continue
            for trecho in fatiar(pagina):
                f.write(trecho.model_dump_json() + "\n")
                total += 1
    console.print(f"[green]{total} trechos gravados em {destino}[/green]")


@app.command("build-index")
def build_index() -> None:
    """Monta a base vetorial (trechos didáticos + questões do banco, sem a bateria)."""
    from estudai.infrastructure.ingestion.indice import montar_documentos
    from estudai.infrastructure.ingestion.wikibooks import Trecho
    from estudai.infrastructure.knowledge.hybrid import HybridKnowledgeBase
    from estudai.infrastructure.knowledge.question_bank import JsonlQuestionBank

    container = _container()
    s = container.settings
    trechos = [
        Trecho.model_validate_json(linha) for linha in s.trechos_path.read_text(encoding="utf-8").splitlines() if linha
    ]
    questoes = JsonlQuestionBank.load(s.questoes_path).find()
    documentos = montar_documentos(trechos, questoes)
    kb = HybridKnowledgeBase.build(documentos, container.embedder, alpha=s.kb_alpha)
    kb.save(s.index_dir)
    console.print(f"[green]{len(kb)} documentos indexados em {s.index_dir}[/green]")


@app.command()
def estudar(
    aluno: Annotated[str, typer.Option(help="Identificador do aluno")],
    topico: Annotated[Topico, typer.Option()],
    ciclos: int = 1,
) -> None:
    """Sessão interativa de tutoria + prática no terminal."""
    from estudai.interfaces.console import ConsoleLearner

    container = _container()
    orquestrador = container.orquestrador()
    learner = ConsoleLearner(console)
    for i in range(1, ciclos + 1):
        console.rule(f"Ciclo {i} · {topico.value}")
        resultado = orquestrador.executar_ciclo(aluno, topico, learner)
        console.print(
            f"Perfil v{resultado.versao_perfil_inicial} → v{resultado.versao_perfil_final} · "
            f"acertos {sum(c.correta for c in resultado.correcoes)}/{len(resultado.correcoes)} · "
            f"custo US$ {resultado.uso.total.cost_usd:.4f} · {resultado.latencia_s:.0f}s"
        )


@app.command()
def perfil(aluno: str) -> None:
    """Mostra o histórico de versões do perfil de um aluno."""
    container = _container()
    for versao in container.perfis().history(aluno):
        topicos = {t.value: (e.nivel.value if e.nivel else "-") for t, e in versao.topicos.items()}
        console.print(f"v{versao.versao} [{versao.motivo}] {versao.atualizado_em:%Y-%m-%d %H:%M} {topicos}")


@app.command()
def experimento(
    saida: Annotated[Path, typer.Option(help="Arquivo JSONL de resultados (retomável)")],
    condicoes: Annotated[list[str], typer.Option()] = ["C1", "C2", "C3"],
    topicos: Annotated[list[Topico], typer.Option()] = list(Topico),
    ciclos: int = 1,
    seed: int = 0,
    sem_avaliar_itens: bool = False,
) -> None:
    """Executa o protocolo experimental da Seção 5.3 com os 3 perfis simulados."""
    from estudai.application.orquestrador import ConfigSessao
    from estudai.application.telemetria import UsageLedger
    from estudai.evaluation.experimento import Condicao, Experimento
    from estudai.evaluation.personas import PERSONAS
    from estudai.infrastructure.persistence.json_profiles import JsonProfileRepository

    container = _container()
    s = container.settings
    ledger_avaliacao = UsageLedger()  # aluno simulado e juízes não contam no custo do sistema
    perfis_raiz = saida.parent / f"{saida.stem}_perfis"

    def perfis_novos(aluno_id: str) -> JsonProfileRepository:
        # cada execução começa com um aluno novo (descarta restos de uma execução interrompida)
        shutil.rmtree(perfis_raiz / aluno_id, ignore_errors=True)
        return JsonProfileRepository(perfis_raiz)

    exp = Experimento(
        fabrica_orquestrador=lambda cfg, fonte, perfis: container.orquestrador(cfg, fonte, perfis),
        fabrica_perfis=perfis_novos,
        banco=container.banco,
        llm_aluno=container.llm(s.student or s.economic, ledger_avaliacao),
        juizes=[container.llm(j, ledger_avaliacao) for j in s.judges],
        config_base=ConfigSessao(
            max_rodadas_tutoria=s.max_rodadas_tutoria, exercicios_por_sessao=s.exercicios_por_sessao
        ),
        ciclos=ciclos,
        seed=seed,
        avaliar_itens=not sem_avaliar_itens,
    )
    exp.executar([Condicao(c) for c in condicoes], list(PERSONAS), topicos, saida, progresso=console.print)
    console.print(f"[green]Resultados em {saida}[/green]")
    console.print(
        f"Custo do sistema: US$ {container.ledger.summary().total.cost_usd:.4f} · "
        f"custo da avaliação: US$ {ledger_avaliacao.summary().total.cost_usd:.4f}"
    )


@app.command()
def relatorio(arquivo: Path, orcamento_usd: Annotated[float | None, typer.Option()] = None) -> None:
    """Agrega um JSONL de resultados: métricas, testes de Wilcoxon, κ e critérios de sucesso."""
    from estudai.evaluation.experimento import carregar_resultados
    from estudai.evaluation.relatorio import gerar_relatorio

    rel = gerar_relatorio(carregar_resultados(arquivo), orcamento_usd=orcamento_usd)

    tabela = Table(title="Métricas por condição")
    colunas = [
        "condicao",
        "execucoes",
        "falhas",
        "ganho_medio",
        "rubrica_media",
        "acuracia_diagnostico",
        "taxa_descarte",
        "validade_entregues",
        "itens_com_defeito_critico",
        "custo_medio_usd",
        "latencia_media_s",
    ]
    for c in colunas:
        tabela.add_column(c)
    for resumo in rel.condicoes:
        tabela.add_row(*[_fmt(getattr(resumo, c)) for c in colunas])
    console.print(tabela)

    for comp in rel.comparacoes:
        console.print(
            f"Wilcoxon {comp.metrica}: {comp.base} vs {comp.alvo} → "
            f"p={comp.teste.get('p_valor', float('nan')):.4f} (n={comp.teste['n']:.0f}, "
            f"mediana Δ={comp.teste['mediana_diferenca']:+.3f})"
        )
    console.print(
        f"κ juízes (rubrica): {_fmt(rel.kappa_juizes_rubrica)} · κ resolutores: {_fmt(rel.kappa_resolutores)}"
    )
    for nome, ok in rel.criterios.items():
        marca = {True: "[green]atingido[/green]", False: "[red]não atingido[/red]", None: "[dim]n/a[/dim]"}[ok]
        console.print(f"Critério {nome}: {marca}")


def _fmt(valor) -> str:
    if isinstance(valor, float):
        return f"{valor:.3f}"
    return "-" if valor is None else str(valor)


def _imprimir_uso(container) -> None:
    total = container.ledger.summary().total
    console.print(
        f"LLM: {total.calls} chamadas · {total.input_tokens}+{total.output_tokens} tokens · US$ {total.cost_usd:.4f}"
    )


if __name__ == "__main__":
    app()
