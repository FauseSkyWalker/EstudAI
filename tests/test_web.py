"""Interface web: a ponte thread/fila (`SessaoWeb`) e a tela Streamlit (via AppTest, sem navegador)."""

import time
from pathlib import Path

import pytest
from test_orquestrador import montar

from estudai.application.progresso import EventoProgresso, TipoProgresso
from estudai.domain.correcao import RespostaAluno
from estudai.domain.curriculum import Topico
from estudai.domain.figura import Figura
from estudai.domain.notificacao import TipoNotificacao
from estudai.domain.questoes import Exercicio
from estudai.domain.tutoria import TurnoTutor
from estudai.infrastructure.sandbox.python_executor import SubprocessPythonExecutor
from estudai.interfaces.web.sessao import EstadoSessao, Fase, Pedido, SessaoWeb, TipoEvento, WebLearner

APP = str(Path(__file__).parents[1] / "src" / "estudai" / "interfaces" / "web" / "app.py")
FINAIS = (EstadoSessao.CONCLUIDA, EstadoSessao.ERRO, EstadoSessao.CANCELADA)


@pytest.fixture(scope="module")
def executor():
    return SubprocessPythonExecutor(timeout_s=30)


def esperar_pedido(sessao: SessaoWeb, timeout: float = 60) -> Pedido | None:
    fim = time.monotonic() + timeout
    while sessao.estado not in FINAIS:
        if (pedido := sessao.learner.pendente) is not None:
            return pedido
        assert time.monotonic() < fim, "a sessão não pediu nada a tempo"
        time.sleep(0.01)
    return None


def test_ciclo_completo_pela_ponte(kb, banco, executor, tmp_path):
    orq, _ = montar(kb, banco, executor, tmp_path)
    sessao = SessaoWeb(orq, "ana", Topico.FUNCOES).iniciar()
    while (pedido := esperar_pedido(sessao)) is not None:
        if pedido.turno is not None:
            sessao.learner.responder("acho que é -b/2a")
        else:
            sessao.learner.responder(RespostaAluno(item_id=pedido.item_id, escolha="A"))

    assert sessao.estado == EstadoSessao.CONCLUIDA, sessao.erro
    assert len(sessao.resultado.correcoes) == 3
    eventos = sessao.learner.eventos()
    fases_dos_itens = [e.fase for e in eventos if e.tipo == TipoEvento.ITEM]
    assert fases_dos_itens == [Fase.DIAGNOSTICO] * 4 + [Fase.PRATICA] * 3
    assert [e.tipo for e in eventos if e.fase == Fase.TUTORIA][:2] == [TipoEvento.TUTOR, TipoEvento.ALUNO]
    assert any(
        e.tipo == TipoEvento.NOTIFICACAO and e.conteudo.tipo == TipoNotificacao.RESUMO for e in eventos
    )  # resumo
    # bastidores: o que cada agente fez fica disponível para a tela
    progresso = sessao.progresso()
    assert {e.etapa for e in progresso if e.tipo == TipoProgresso.ETAPA} >= {"tutor", "practice"}
    assert any(e.agente == "Tutor" and e.tipo == TipoProgresso.PENSANDO for e in progresso)


def test_encerrar_sessao_libera_a_thread(kb, banco, executor, tmp_path):
    orq, _ = montar(kb, banco, executor, tmp_path)
    sessao = SessaoWeb(orq, "bia", Topico.FUNCOES).iniciar()
    assert esperar_pedido(sessao) is not None
    sessao.cancelar()
    sessao.aguardar(timeout=10)
    assert sessao.estado == EstadoSessao.CANCELADA


def test_responder_sem_pedido_e_erro():
    with pytest.raises(RuntimeError):
        WebLearner().responder("oi")


# --- tela (Streamlit AppTest) -------------------------------------------------------------


class SessaoFalsa:
    """Sessão parada num pedido, para testar a tela sem rodar o orquestrador."""

    def __init__(self, pedido: Pedido) -> None:
        self.aluno_id, self.topico, self.learner = "ana", Topico.FUNCOES, WebLearner()
        self.learner._pendente = pedido
        self.resultado = self.erro = None
        self.eventos_progresso = []

    @property
    def estado(self) -> EstadoSessao:
        return EstadoSessao.AGUARDANDO_ALUNO if self.learner.pendente else EstadoSessao.TRABALHANDO

    def cancelar(self) -> None: ...

    def progresso(self) -> list:
        return self.eventos_progresso


@pytest.fixture
def app(tmp_path, monkeypatch):
    streamlit = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("ESTUDAI_DATA_DIR", str(tmp_path))  # sem banco nem índice
    streamlit.cache_resource.clear()
    return AppTest.from_file(APP, default_timeout=60)


def test_tela_avisa_quando_faltam_dados(app):
    app.run()
    assert not app.exception
    erros = " ".join(e.value for e in app.sidebar.error)
    assert "ingest-questions" in erros and "build-index" in erros


def test_tela_mostra_questao_e_envia_resposta(app):
    item = Exercicio(
        id="e1", topico=Topico.FUNCOES, enunciado="Quanto é f(2) para f(x) = x²?",
        alternativas=["2", "4", "6", "8", "10"], gabarito="B",
    )  # fmt: skip
    sessao = SessaoFalsa(Pedido(fase=Fase.PRATICA, item_id="e1", item=item))
    app.session_state["sessao"] = sessao
    app.run()
    assert not app.exception
    assert app.radio[0].options == ["A) 2", "B) 4", "C) 6", "D) 8", "E) 10"]

    app.radio[0].set_value("B")
    app.text_area[0].set_value("2 ao quadrado")
    next(b for b in app.button if b.label == "Enviar resposta").click().run()
    resposta = sessao.learner._respostas.get_nowait()
    assert (resposta.item_id, resposta.escolha, resposta.resolucao) == ("e1", "B", "2 ao quadrado")


def test_tela_envia_resposta_ao_tutor(app):
    turno = TurnoTutor(conceito_alvo="funcao_quadratica", explicacao="O vértice...", pergunta_verificacao="Onde fica?")
    sessao = SessaoFalsa(Pedido(fase=Fase.TUTORIA, turno=turno))
    app.session_state["sessao"] = sessao
    app.run()
    app.chat_input[0].set_value("em -b/2a").run()
    assert sessao.learner._respostas.get_nowait() == "em -b/2a"


PNG_1PX = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def test_tela_mostra_figura_do_tutor_e_bastidores(app):
    turno = TurnoTutor(
        conceito_alvo="funcao_quadratica",
        explicacao="Veja a parábola.",
        pergunta_verificacao="Onde ficam as raízes?",
        imagem=Figura(descricao="Parábola com raízes 1 e 3", png_base64=PNG_1PX),
    )
    sessao = SessaoFalsa(Pedido(fase=Fase.TUTORIA, turno=turno))
    sessao.learner._registrar(TipoEvento.TUTOR, turno)
    sessao.eventos_progresso = [
        EventoProgresso(tipo=TipoProgresso.ETAPA, descricao="Tutoria", etapa="tutor"),
        EventoProgresso(tipo=TipoProgresso.PENSANDO, descricao="está pensando", etapa="tutor", agente="Tutor"),
    ]
    app.session_state["sessao"] = sessao
    app.run()
    assert not app.exception
    assert len(app.get("image")) == 1
    legendas = " ".join(c.value for c in app.caption)
    assert "▶️ **Tutoria**" in legendas and "◻️ Relatório" in legendas  # faixa do pipeline
    assert any("Bastidores" in e.label for e in app.expander)
