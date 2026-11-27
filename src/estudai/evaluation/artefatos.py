import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from estudai.application.telemetria import UsageRecord
from estudai.domain.mensagens import Mensagem
from estudai.evaluation.registros import RegistroCiclo, ResultadoExecucao


def diretorio_mensagens(saida: Path) -> Path:
    """`results/exp1.jsonl` -> `results/exp1_mensagens/`."""
    return saida.with_name(f"{saida.stem}_mensagens")


def diretorio_usage(saida: Path) -> Path:
    """`results/exp1.jsonl` -> `results/exp1_usage/`."""
    return saida.with_name(f"{saida.stem}_usage")


def salvar_snapshot(
    diretorio: Path | None,
    raiz: Path | None,
    nome: str,
    registros: Iterable[BaseModel],
) -> str | None:
    """Grava um JSONL por ciclo e devolve o caminho relativo à raiz dos resultados."""
    if diretorio is None or raiz is None:  # `executar_um` chamado fora de `executar`
        return None
    diretorio.mkdir(parents=True, exist_ok=True)
    arquivo = diretorio / f"{nome}.jsonl"
    arquivo.write_text("".join(registro.model_dump_json() + "\n" for registro in registros), encoding="utf-8")
    return arquivo.relative_to(raiz).as_posix()


def anexar(caminho: Path, resultado: ResultadoExecucao) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("a", encoding="utf-8") as f:
        f.write(resultado.model_dump_json() + "\n")


def regravar(caminho: Path, resultados: list[ResultadoExecucao]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text("".join(r.model_dump_json() + "\n" for r in resultados), encoding="utf-8")


def carregar_resultados(caminho: Path) -> list[ResultadoExecucao]:
    with caminho.open(encoding="utf-8") as f:
        return [ResultadoExecucao.model_validate(json.loads(linha)) for linha in f if linha.strip()]


def carregar_mensagens(saida: Path, registro: RegistroCiclo) -> list[Mensagem]:
    """Mensagens do protocolo de um ciclo, para auditar o que cada agente enviou e recebeu."""
    if registro.arquivo_mensagens is None:
        return []
    arquivo = saida.parent / registro.arquivo_mensagens
    return [Mensagem.model_validate_json(linha) for linha in arquivo.read_text(encoding="utf-8").splitlines() if linha]


def carregar_usage(saida: Path, registro: RegistroCiclo) -> list[UsageRecord]:
    """Chamadas individuais de LLM do ciclo (sistema + avaliação), para custo por chamada."""
    if registro.arquivo_usage is None:
        return []
    arquivo = saida.parent / registro.arquivo_usage
    linhas = arquivo.read_text(encoding="utf-8").splitlines()
    return [UsageRecord.model_validate_json(linha) for linha in linhas if linha]
