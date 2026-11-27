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
from dotenv import load_dotenv
from rich.console import Console

from estudai.domain.curriculum import Topico
from estudai.domain.mensagens import Papel

# telemetria opcional (ex.: LANGSMITH_*) configurada no .env precisa estar no ambiente do processo
load_dotenv()

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


def _exigir_dados(container) -> None:
    """Falha cedo, antes de gastar chamadas de LLM, se a ingestão ou o índice não foram feitos."""
    faltando = container.dados_faltando()
    for caminho, comando in faltando:
        console.print(f"[red]{caminho} não existe.[/red] Rode antes: [bold]{comando}[/bold]")
    if faltando:
        raise typer.Exit(code=1)


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
        ClassificadorQuestoes(container.classificador()),
        RotuloCache(s.rotulos_cache_path),
        bateria_por_topico=bateria,
        workers=workers,
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
    from estudai.infrastructure.knowledge.database import KnowledgeDatabase
    from estudai.infrastructure.knowledge.question_bank import JsonlQuestionBank

    container = _container()
    s = container.settings

    trechos = [
        Trecho.model_validate_json(linha) for linha in s.trechos_path.read_text(encoding="utf-8").splitlines() if linha
    ]
    questoes = JsonlQuestionBank.load(s.questoes_path).find()

    documentos = montar_documentos(trechos, questoes)

    kb = KnowledgeDatabase.build(s.knowledge_db_uri, documentos, container.embedder)

    console.print(f"[green]{len(kb)} documentos indexados em {s.knowledge_db_uri}[/green]")


@app.command()
def estudar(
    aluno: Annotated[str, typer.Option(help="Identificador do aluno")],
    topico: Annotated[Topico, typer.Option()],
    ciclos: int = 1,
    bastidores: Annotated[bool, typer.Option(help="Mostra o que cada agente está fazendo")] = True,
) -> None:
    """Sessão interativa de tutoria + prática no terminal."""
    from estudai.interfaces.console import ConsoleLearner

    container = _container()
    _exigir_dados(container)
    orquestrador = container.orquestrador()
    learner = ConsoleLearner(console, pasta_figuras=container.settings.data_dir / "figuras" / aluno)
    for i in range(1, ciclos + 1):
        console.rule(f"Ciclo {i} · {topico.value}")
        observador = learner.bastidores if bastidores else None
        resultado = orquestrador.executar_ciclo(aluno, topico, learner, observador=observador)
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
    condicoes: Annotated[
        list[str], typer.Option(help="Condições a serem aplicadas, sequencialmente: --condicoes C1 --condicoes C2")
    ] = ["C1", "C2", "C3"],
    topicos: Annotated[list[Topico], typer.Option()] = list(Topico),
    personas: Annotated[list[str] | None, typer.Option(help="IDs de persona (vazio = todas)")] = None,
    ciclos: Annotated[int, typer.Option(help="Ciclos por execução")] = 2,
    exercicios_por_sessao: Annotated[int, typer.Option(help="Itens de prática por ciclo")] = 4,
    conceitos_por_sessao: Annotated[int, typer.Option(help="Conceitos-alvo ensinados por ciclo")] = 2,
    seed: int = 0,
    sem_avaliar_itens: bool = False,
    refazer_falhas: Annotated[bool, typer.Option(help="Reexecuta entradas com erro e regrava o JSONL")] = False,
    max_custo_usd: Annotated[float | None, typer.Option(help="Interrompe execução ao atingir custo")] = None,
) -> None:
    """Executa o protocolo experimental da Seção 5.3 com os 3 perfis simulados."""
    from estudai.application.cycle.config import ConfigSessao
    from estudai.application.telemetria import UsageLedger
    from estudai.evaluation.experimento import Experimento
    from estudai.evaluation.personas import PERSONAS
    from estudai.evaluation.registros import Condicao
    from estudai.infrastructure.persistence.json_profiles import JsonProfileRepository

    container = _container()
    _exigir_dados(container)
    s = container.settings
    ledger_avaliacao = UsageLedger()  # aluno simulado e juízes não contam no custo do sistema
    perfis_raiz = saida.parent / f"{saida.stem}_perfis"
    personas_selecionadas = [p for p in PERSONAS if not personas or p.id in personas]
    if not personas_selecionadas:
        console.print("[red]nenhuma persona selecionada[/red]")
        raise typer.Exit(code=1)
    condicoes_enum = [Condicao(c) for c in condicoes]

    def perfis_novos(aluno_id: str) -> JsonProfileRepository:
        # cada execução começa com um aluno novo (descarta restos de uma execução interrompida)
        shutil.rmtree(perfis_raiz / aluno_id, ignore_errors=True)
        return JsonProfileRepository(perfis_raiz)

    exp = Experimento(
        fabrica_orquestrador=lambda cfg, fonte, perfis, on_tutoring=None: container.orquestrador(
            cfg, fonte, perfis, com_figuras=False, on_tutoring=on_tutoring
        ),
        fabrica_perfis=perfis_novos,
        banco=container.banco,
        llm_aluno=container.chat(s.student or s.economic, Papel.ALUNO_SIMULADO, ledger_avaliacao),
        juizes=[container.chat(j, Papel.JUIZ, ledger_avaliacao) for j in s.judges],
        config_base=ConfigSessao(
            max_rodadas_tutoria=s.max_rodadas_tutoria,
            exercicios_por_sessao=exercicios_por_sessao,
            conceitos_por_sessao=conceitos_por_sessao,
        ),
        ciclos=ciclos,
        seed=seed,
        avaliar_itens=not sem_avaliar_itens,
        ledger_sistema=container.ledger,
        ledger_avaliacao=ledger_avaliacao,
    )
    exp.executar(
        condicoes_enum,
        personas_selecionadas,
        topicos,
        saida,
        progresso=console.print,
        refazer_falhas=refazer_falhas,
        custo_atual=lambda: container.ledger.summary().total.cost_usd + ledger_avaliacao.summary().total.cost_usd,
        max_custo_usd=max_custo_usd,
    )
    console.print(f"[green]Resultados em {saida}[/green]")
    console.print(
        f"Custo do sistema: US$ {container.ledger.summary().total.cost_usd:.4f} · "
        f"custo da avaliação: US$ {ledger_avaliacao.summary().total.cost_usd:.4f}"
    )


@app.command()
def relatorio(
    arquivo: Path,
    orcamento_usd: Annotated[float | None, typer.Option()] = None,
    saida: Annotated[Path | None, typer.Option(help="Prefixo para exportar .json, .md e .csv")] = None,
) -> None:
    from estudai.evaluation.artefatos import carregar_resultados
    from estudai.evaluation.exportacao import exportar
    from estudai.evaluation.relatorio import gerar_relatorio
    from estudai.interfaces.relatorio_view import imprimir_relatorio

    resultados = carregar_resultados(arquivo)
    rel = gerar_relatorio(resultados, orcamento_usd=orcamento_usd)
    imprimir_relatorio(rel, console)

    if saida is not None:
        exportar(rel, resultados, saida)
        console.print(f"[green]Relatório exportado com prefixo {saida}[/green]")


@app.command()
def web(porta: int = 8501) -> None:
    """Abre a interface web (Streamlit) da sessão de estudo no navegador."""
    import importlib.util
    import subprocess

    if importlib.util.find_spec("streamlit") is None:
        console.print(
            "[red]Streamlit não instalado.[/red] Rode: [bold]uv sync --extra ingest --extra web[/bold] "
            "(o uv sync remove os extras não listados)"
        )
        raise typer.Exit(code=1)
    app_web = Path(__file__).with_name("web") / "app.py"
    raise typer.Exit(
        subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_web), "--server.port", str(porta)])
    )


@app.command("verificar-modelos")
def verificar_modelos() -> None:
    """Faz uma chamada estruturada mínima em cada modelo configurado (econômico, fronteira, painel,
    juízes, aluno simulado) para validar chaves, ids e saída estruturada antes de rodar o pipeline."""
    import time

    from pydantic import BaseModel

    from estudai.agents.base import run_config
    from estudai.application.telemetria import UsageLedger

    class Resposta(BaseModel):
        resultado: int

    container = _container()
    s = container.settings
    papeis = [("economic", s.economic), ("frontier", s.frontier), *[("panel", c) for c in s.panel or []]]
    papeis += [("judge", j) for j in s.judges] + ([("student", s.student)] if s.student else [])
    vistos, falhas = set(), 0
    for nome, cfg in papeis:
        chave = (cfg.provider, cfg.model, cfg.base_url)
        if chave in vistos:
            continue
        vistos.add(chave)
        ledger, inicio = UsageLedger(), time.perf_counter()
        try:
            modelo = container.chat(cfg, Papel.CLASSIFICADOR, ledger=ledger)
            r = modelo.with_structured_output(Resposta).invoke("Quanto é 6 * 7?", config=run_config("verificacao"))
            ok = r.resultado == 42
            total = ledger.summary().total
            detalhe = f"{time.perf_counter() - inicio:.1f}s · US$ {total.cost_usd:.4f}" + (
                "" if ok else f" · respondeu {r}"
            )
        except Exception as exc:  # noqa: BLE001 - diagnóstico: mostra qualquer falha
            ok, detalhe = False, f"{type(exc).__name__}: {str(exc)[:200]}"
        falhas += not ok
        marca = "[green]OK[/green]" if ok else "[red]FALHOU[/red]"
        console.print(f"{marca} {nome:9} {cfg.provider}/{cfg.model}  {detalhe}")

    try:
        from estudai.ports.knowledge import SearchMode

        hits = container.kb.search("função quadrática", SearchMode.HIBRIDA, k=1, topico=Topico.FUNCOES)
        if not hits:
            raise RuntimeError("nenhum resultado na base de conhecimento")
        console.print(f"[green]OK[/green] knowledge base + embedder  ({hits[0].document.id})")
    except Exception as exc:  # noqa: BLE001 - diagnóstico
        falhas += 1
        console.print(f"[red]FALHOU[/red] knowledge base + embedder  {type(exc).__name__}: {str(exc)[:200]}")

    if falhas:
        raise typer.Exit(code=1)


def _imprimir_uso(container) -> None:
    total = container.ledger.summary().total
    console.print(
        f"LLM: {total.calls} chamadas · {total.input_tokens}+{total.output_tokens} tokens · US$ {total.cost_usd:.4f}"
    )


if __name__ == "__main__":
    app()
