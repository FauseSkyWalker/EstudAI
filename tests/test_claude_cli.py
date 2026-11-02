"""Adaptador do Claude Code CLI, exercitado contra um CLI falso (`fake_claude_cli.py`)."""

import json
import os
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from estudai.agents.base import run_config
from estudai.application.telemetria import UsageLedger
from estudai.infrastructure.llm import claude_cli
from estudai.infrastructure.llm.callbacks import UsageLedgerCallback
from estudai.infrastructure.llm.claude_cli import ClaudeCLIChatModel, ClaudeCLIError, localizar_claude_cli
from estudai.infrastructure.llm.factory import build_chat_model

FAKE = str(Path(__file__).with_name("fake_claude_cli.py"))


class R(BaseModel):
    resultado: int


@pytest.fixture
def log(tmp_path, monkeypatch) -> Path:
    caminho = tmp_path / "chamadas.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(caminho))
    return caminho


def chamadas(log: Path) -> list[dict]:
    return [json.loads(linha) for linha in log.read_text(encoding="utf-8").splitlines()]


def modelo(ledger: UsageLedger | None = None, **kw) -> ClaudeCLIChatModel:
    callbacks = [UsageLedgerCallback(ledger, model="claude-haiku-5-5")] if ledger else []
    return ClaudeCLIChatModel(
        model="claude-haiku-5-5", command=[sys.executable, FAKE], callbacks=callbacks, **{"max_retries": 0, **kw}
    )


def test_saida_estruturada_com_tokens_e_custo_reais(log):
    ledger = UsageLedger()
    r = modelo(ledger).with_structured_output(R).invoke("6 * 7?", config=run_config("tutor"))
    assert r == R(resultado=42)

    args = chamadas(log)[0]["args"]
    schema = json.loads(args[args.index("--json-schema") + 1])
    assert schema["properties"]["resultado"]["type"] == "integer"
    resumo = ledger.summary()
    assert resumo.por_agente["tutor"].calls == 1
    assert resumo.total.cost_usd == pytest.approx(0.0123)  # custo informado pelo CLI, não pela tabela
    assert (resumo.total.input_tokens, resumo.total.output_tokens) == (150, 20)


def test_isola_o_cli_e_envia_system_prompt_por_arquivo(log):
    msgs = [SystemMessage("SOU O TUTOR"), HumanMessage("p1"), AIMessage("a1"), HumanMessage("p2")]
    assert modelo().invoke(msgs).content == "resposta livre"

    chamada = chamadas(log)[0]
    assert chamada["system"] == "SOU O TUTOR"
    assert "[ASSISTENTE]\na1" in chamada["prompt"] and chamada["prompt"].rstrip().endswith("p2")
    args = chamada["args"]
    assert args[args.index("--tools") + 1] == "" and "--no-session-persistence" in args
    assert "--json-schema" not in args
    assert "estudai-claude-" in chamada["cwd"]  # diretório temporário: sem CLAUDE.md do projeto


def test_saida_invalida_volta_como_erro_de_parsing_para_o_reparo(log, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_STRUCTURED", '{"resultado": "não é número"}')
    out = modelo().with_structured_output(R, include_raw=True).invoke("?")
    assert out["parsed"] is None and out["parsing_error"] is not None


def test_erro_definitivo_nao_repete(log, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "erro_definitivo")
    with pytest.raises(ClaudeCLIError, match="400"):
        modelo(max_retries=3).invoke("?")
    assert len(chamadas(log)) == 1


def test_erro_transiente_repete(log, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "erro_transiente")
    monkeypatch.setattr(claude_cli.time, "sleep", lambda _: None)
    with pytest.raises(ClaudeCLIError, match="529"):
        modelo(max_retries=2).invoke("?")
    assert len(chamadas(log)) == 3


def test_saida_nao_json_vira_erro(log, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "lixo")
    with pytest.raises(ClaudeCLIError, match="não-JSON"):
        modelo().invoke("?")


@pytest.mark.skipif(os.name != "nt", reason="o desvio do .cmd só existe no Windows")
def test_no_windows_evita_o_cmd_e_usa_o_executavel_nativo(tmp_path, monkeypatch):
    nativo = tmp_path / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    nativo.parent.mkdir(parents=True)
    nativo.touch()
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _: str(tmp_path / "claude.CMD"))
    assert localizar_claude_cli() == [str(nativo)]


def test_factory_monta_claude_cli_sem_chave():
    m = build_chat_model(
        provider="claude_cli", model="claude-sonnet-5-5", api_key=None, temperature=0.7,
        max_retries=1, timeout_s=60, effort="low",
    )  # fmt: skip
    assert isinstance(m, ClaudeCLIChatModel) and (m.model, m.effort, m.max_retries) == ("claude-sonnet-5-5", "low", 1)


def test_bootstrap_nao_exige_chave_para_claude_cli():
    from estudai.bootstrap import Container
    from estudai.domain.mensagens import Papel
    from estudai.settings import ModelConfig, Settings

    cli = ModelConfig(provider="claude_cli", model="claude-haiku-5-5")
    s = Settings(_env_file=None, economic=cli, frontier=cli)
    assert isinstance(Container(s).economico(Papel.RESUMO), ClaudeCLIChatModel)
