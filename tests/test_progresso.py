"""Bastidores: o callback de progresso enxerga etapas, agentes, ferramentas e figuras do ciclo."""

import threading

import pytest
from fakes import FIGURA, RendererFalso, ScriptedLearner
from test_orquestrador import T, montar

from estudai.application.progresso import ETAPAS, EventoProgresso, ProgressoCallback, TipoProgresso
from estudai.infrastructure.sandbox.python_executor import SubprocessPythonExecutor


@pytest.fixture(scope="module")
def executor():
    return SubprocessPythonExecutor(timeout_s=30)


@pytest.fixture
def eventos(kb, banco, executor, tmp_path) -> list[EventoProgresso]:
    """Um ciclo completo roteirizado (com figura), observado do começo ao fim."""
    orq, _ = montar(kb, banco, executor, tmp_path, renderer=RendererFalso(), figuras=[FIGURA])
    registrados: list[EventoProgresso] = []
    lock = threading.Lock()

    def observador(evento: EventoProgresso) -> None:
        with lock:  # o painel de resolutores emite de várias threads
            registrados.append(evento)

    orq.executar_ciclo("ana", T, ScriptedLearner(padrao="certa"), observador=observador)
    return registrados


def _de(eventos, tipo):
    return [e for e in eventos if e.tipo == tipo]


def _etapas_aparecem_na_ordem_do_pipeline(eventos):
    etapas = [e.etapa for e in _de(eventos, TipoProgresso.ETAPA)]
    ordem = list(ETAPAS)
    assert etapas == sorted(etapas, key=ordem.index)  # sem repetição nem volta
    assert {"load_profile", "diagnose", "tutor", "practice", "record_episode"} <= set(etapas)


def _cada_agente_aparece_pensando_na_sua_etapa(eventos):
    pensando = _de(eventos, TipoProgresso.PENSANDO)
    agentes = {e.agente for e in pensando}
    assert {"Diagnosticador", "Tutor", "Agente de Conteúdo", "Corretor", "Agente de Resumo"} <= agentes
    assert all(e.etapa == "tutor" for e in pensando if e.agente == "Tutor")
    # toda chamada iniciada é concluída
    assert len(_de(eventos, TipoProgresso.RESPONDEU)) == len(pensando)


def _painel_de_resolutores_em_threads_tambem_e_visto(eventos):
    assert any(e.agente == "Painel de resolutores" for e in _de(eventos, TipoProgresso.PENSANDO))


def _ferramentas_e_figuras_sao_descritas(eventos):
    ferramentas = [e.descricao for e in _de(eventos, TipoProgresso.FERRAMENTA)]
    assert any("busca hibrida" in d and "função quadrática vértice" in d for d in ferramentas)
    figuras = _de(eventos, TipoProgresso.FIGURA)
    assert [(e.agente, e.etapa) for e in figuras] == [("Tutor", "tutor")]
    assert "Reta identidade" in figuras[0].descricao


def test_bastidores_de_um_ciclo_completo(eventos):
    _etapas_aparecem_na_ordem_do_pipeline(eventos)
    _cada_agente_aparece_pensando_na_sua_etapa(eventos)
    _painel_de_resolutores_em_threads_tambem_e_visto(eventos)
    _ferramentas_e_figuras_sao_descritas(eventos)


def test_observador_com_defeito_nao_derruba_o_ciclo(kb, banco, executor, tmp_path):
    orq, _ = montar(kb, banco, executor, tmp_path)

    def quebrado(evento):
        raise RuntimeError("bug na tela")

    r = orq.executar_ciclo("bia", T, ScriptedLearner(padrao="certa"), observador=quebrado)
    assert r.correcoes


def test_callback_ignora_nos_internos_do_agente_de_ferramentas():
    vistos = []
    cb = ProgressoCallback(vistos.append)
    cb.on_chain_start({}, {}, run_id=1, metadata={"langgraph_node": "tutor"}, name="tutor")
    cb.on_chain_start({}, {}, run_id=2, metadata={"langgraph_node": "model"}, name="model")
    cb.on_chain_start({}, {}, run_id=3, metadata={"langgraph_node": "tutor"}, name="RunnableSequence")
    assert [e.descricao for e in vistos] == ["Tutoria"]


def test_cada_figura_emite_um_evento_de_progresso(kb, banco, executor, tmp_path):
    segunda = {"codigo": "plt.plot([1, 0], [0, 1])", "descricao": "Outra reta"}
    orq, _ = montar(kb, banco, executor, tmp_path, renderer=RendererFalso(), figuras=[FIGURA, segunda])
    registrados: list[EventoProgresso] = []

    orq.executar_ciclo("ana", T, ScriptedLearner(padrao="certa"), observador=registrados.append)

    figuras = [e for e in registrados if e.tipo == TipoProgresso.FIGURA]
    assert len(figuras) == 2
    assert "Reta identidade" in figuras[0].descricao and "Outra reta" in figuras[1].descricao


def test_figuras_de_questoes_emitem_progresso(kb, banco, executor, tmp_path):
    orq, _ = montar(kb, banco, executor, tmp_path, renderer=RendererFalso(), figuras_item=[FIGURA])
    registrados: list[EventoProgresso] = []

    orq.executar_ciclo("ana", T, ScriptedLearner(padrao="certa"), observador=registrados.append)

    figuras = [e for e in registrados if e.tipo == TipoProgresso.FIGURA]
    assert any(e.agente == "Gerador de Exercícios" and "questão" in e.descricao for e in figuras)
    notas = [e for e in registrados if e.tipo == TipoProgresso.NOTA]
    assert any("pediram figura" in e.descricao for e in notas)
