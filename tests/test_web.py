"""Interface web: a ponte thread/fila (`SessaoWeb`) e a tela Streamlit (via AppTest, sem navegador)."""

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_orquestrador import montar

from estudai.application.orquestrador import ResultadoCiclo
from estudai.application.pratica import ResultadoFornecimento
from estudai.application.progresso import EventoProgresso, TipoProgresso
from estudai.application.sessoes import (
    EventoSessao,
    Fase,
    ItemReplay,
    SessaoRegistrada,
    TipoEvento,
)
from estudai.application.telemetria import UsageSummary
from estudai.domain.correcao import RespostaAluno
from estudai.domain.curriculum import Topico
from estudai.domain.figura import Figura, FormatoFigura
from estudai.domain.material import PacoteMaterial
from estudai.domain.notificacao import Notificacao, TipoNotificacao
from estudai.domain.planejamento import ObjetivoSessao
from estudai.domain.questoes import Exercicio
from estudai.domain.tutoria import Encerramento, TranscricaoTutoria, TurnoTutor
from estudai.infrastructure.persistence.json_sessions import JsonSessionRepository
from estudai.infrastructure.sandbox.python_executor import SubprocessPythonExecutor
from estudai.interfaces.web.sessao import EstadoSessao, Pedido, SessaoWeb, WebLearner

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
    # cada mensagem leva o trace que a produziu; o que sobra fica como trace pendente
    assert any(e.trilha for e in eventos)
    anexados = sum(len(e.trilha) for e in eventos)
    assert anexados + len(sessao.progresso_pendente()) == len(progresso)


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


def test_web_learner_anexa_o_progresso_a_mensagem():
    progresso: list[EventoProgresso] = []
    learner = WebLearner(lambda: progresso)
    progresso.append(EventoProgresso(tipo=TipoProgresso.ETAPA, descricao="Tutoria", etapa="tutor"))

    learner.notify(Notificacao(tipo=TipoNotificacao.AVISO, texto="oi"))

    evento = learner.eventos()[0]
    assert [e.descricao for e in evento.trilha] == ["Tutoria"]
    assert learner.progresso_consumido == 1


# --- tela (Streamlit AppTest) -------------------------------------------------------------


class SessaoFalsa:
    """Sessão parada num pedido, para testar a tela sem rodar o orquestrador."""

    def __init__(self, pedido: Pedido) -> None:
        self.id = "teste"
        self.aluno_id, self.topico = "ana", Topico.FUNCOES
        self.eventos_progresso = []
        self.learner = WebLearner(lambda: self.eventos_progresso)
        self.learner._pendente = pedido
        self.resultado = self.erro = None

    @property
    def estado(self) -> EstadoSessao:
        return EstadoSessao.AGUARDANDO_ALUNO if self.learner.pendente else EstadoSessao.TRABALHANDO

    def cancelar(self) -> None: ...

    def progresso(self) -> list:
        return self.eventos_progresso

    def progresso_pendente(self) -> list:
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
        imagens=[Figura(descricao="Parábola com raízes 1 e 3", formato=FormatoFigura.PNG, conteudo=PNG_1PX)],
    )
    sessao = SessaoFalsa(Pedido(fase=Fase.TUTORIA, turno=turno))
    sessao.eventos_progresso = [
        EventoProgresso(tipo=TipoProgresso.ETAPA, descricao="Tutoria", etapa="tutor"),
        EventoProgresso(tipo=TipoProgresso.PENSANDO, descricao="está pensando", etapa="tutor", agente="Tutor"),
    ]
    sessao.learner._registrar(TipoEvento.TUTOR, turno)
    app.session_state["sessao"] = sessao
    app.run()
    assert not app.exception
    assert len(app.get("image")) == 1
    assert any("Bastidores desta mensagem" in e.label for e in app.expander)


def test_desenhar_figura_despacha_png_e_mermaid(monkeypatch):
    from estudai.interfaces.web import figuras as web_figuras

    chamadas = []
    monkeypatch.setattr(web_figuras.st, "image", lambda *a, **k: chamadas.append(("image", a, k)))
    monkeypatch.setattr(web_figuras.st, "mermaid_chart", lambda *a, **k: chamadas.append(("mermaid", a, k)))

    web_figuras.desenhar_figura(Figura(descricao="Parábola", formato=FormatoFigura.PNG, conteudo=PNG_1PX))
    web_figuras.desenhar_figura(
        Figura(
            descricao="Fluxo",
            formato=FormatoFigura.MERMAID,
            conteudo="flowchart TD\n  A --> B",
        )
    )

    assert chamadas[0][0] == "image" and chamadas[0][2]["caption"] == "Parábola"
    assert chamadas[1][0] == "mermaid" and chamadas[1][1][0].startswith("flowchart TD")
    assert chamadas[1][2]["alt"] == "Fluxo"


def test_tela_aceita_figura_mermaid(app):
    turno = TurnoTutor(
        conceito_alvo="funcao_afim",
        explicacao="Veja o fluxo.",
        pergunta_verificacao="Qual é o próximo passo?",
        imagens=[
            Figura(
                descricao="Fluxo da função afim",
                formato=FormatoFigura.MERMAID,
                conteudo="flowchart TD\n  A[Função afim] --> B[Gráfico: reta]",
            )
        ],
    )
    sessao = SessaoFalsa(Pedido(fase=Fase.TUTORIA, turno=turno))
    sessao.learner._registrar(TipoEvento.TUTOR, turno)
    app.session_state["sessao"] = sessao
    app.run()
    assert not app.exception


def test_tela_mostra_figuras_da_questao(app):
    item = Exercicio(
        id="e1",
        topico=Topico.FUNCOES,
        enunciado="Veja o gráfico e responda.",
        alternativas=["2", "4", "6", "8", "10"],
        gabarito="B",
        descricoes_imagens=["Gráfico de f(x) = x²"],
        imagens=[
            Figura(
                descricao="Gráfico de f(x) = x²",
                formato=FormatoFigura.PNG,
                conteudo=PNG_1PX,
            )
        ],
    )
    sessao = SessaoFalsa(Pedido(fase=Fase.PRATICA, item_id="e1", item=item))
    app.session_state["sessao"] = sessao
    app.run()
    assert not app.exception
    assert len(app.get("image")) == 1


def test_quadro_ao_vivo_nao_usa_status(monkeypatch):
    from estudai.interfaces.web import bastidores

    monkeypatch.setattr(bastidores.st, "status", lambda *a, **k: pytest.fail("st.status não deve ser usado"))
    legendas: list[str] = []
    monkeypatch.setattr(bastidores.st, "caption", lambda texto: legendas.append(texto))

    bastidores.quadro_ao_vivo([])

    assert "Preparando o ciclo" in legendas[-1]


def test_tela_mantem_campo_do_tutor_visivel_enquanto_agentes_trabalham(app):
    turno = TurnoTutor(conceito_alvo="funcao_quadratica", explicacao="O vértice...", pergunta_verificacao="Onde fica?")
    sessao = SessaoFalsa(Pedido(fase=Fase.TUTORIA, turno=turno))
    sessao.learner._pendente = None  # os agentes estão trabalhando
    sessao.eventos_progresso = [
        EventoProgresso(tipo=TipoProgresso.ETAPA, descricao="Tutoria", etapa="tutor"),
        EventoProgresso(tipo=TipoProgresso.PENSANDO, descricao="está pensando", etapa="tutor", agente="Tutor"),
    ]
    app.session_state["sessao"] = sessao

    app.run()

    assert not app.exception
    assert app.chat_input and app.chat_input[0].disabled


def test_quadro_ao_vivo_mostra_todo_o_progresso(monkeypatch):
    from estudai.interfaces.web import bastidores

    class Caixa:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    linhas: list[str] = []
    monkeypatch.setattr(bastidores.st, "container", lambda **kwargs: Caixa())
    monkeypatch.setattr(bastidores.st, "markdown", lambda texto: linhas.append(texto))
    eventos = [
        EventoProgresso(tipo=TipoProgresso.PENSANDO, descricao=f"passo {i}", etapa="tutor", agente="Tutor")
        for i in range(12)
    ]

    bastidores.quadro_ao_vivo(eventos)

    texto = "\n".join(linhas)
    assert all(f"passo {i}" in texto for i in range(12))


def _resultado_falso(aluno: str = "ana", topico: Topico = Topico.FUNCOES) -> ResultadoCiclo:
    objetivo = ObjetivoSessao(topico=topico, conceitos_alvo=["funcao_afim"], nivel_aluno=None, descricao="teste")
    return ResultadoCiclo(
        aluno_id=aluno,
        topico=topico,
        adaptativo=True,
        versao_perfil_inicial=0,
        versao_perfil_final=1,
        sondagem=[],
        diagnostico_inicial=None,
        objetivo=objetivo,
        material=PacoteMaterial(pedido="material"),
        transcricao=TranscricaoTutoria(objetivo=objetivo, encerramento=Encerramento.OBJETIVO_ATINGIDO),
        resumo=None,
        fornecimento=ResultadoFornecimento(),
        correcoes=[],
        removidos=[],
        diagnostico_final=None,
        relatorio=None,
        mensagens=[],
        uso=UsageSummary(),
        latencia_s=0.0,
    )


def _registro_falso(
    aluno: str = "ana",
    topico: Topico = Topico.FUNCOES,
    finalizada_em: datetime | None = None,
) -> SessaoRegistrada:
    imagem = Figura(descricao="Parábola", formato=FormatoFigura.PNG, conteudo=PNG_1PX)
    turno = TurnoTutor(
        conceito_alvo="funcao_afim",
        explicacao="Mensagem do tutor",
        pergunta_verificacao="Qual é o coeficiente?",
        imagens=[imagem],
    )
    item = Exercicio(
        id="e1",
        topico=topico,
        enunciado="Quanto é f(2)?",
        alternativas=["2", "4", "6", "8", "10"],
        gabarito="B",
        imagens=[imagem],
    )
    return SessaoRegistrada(
        aluno_id=aluno,
        topico=topico,
        iniciada_em=datetime(2026, 1, 1, 10, tzinfo=UTC),
        finalizada_em=finalizada_em or datetime(2026, 1, 1, 11, tzinfo=UTC),
        resultado=_resultado_falso(aluno, topico),
        eventos=[
            EventoSessao(fase=Fase.TUTORIA, tipo=TipoEvento.TUTOR, turno=turno, turno_imagens=[imagem]),
            EventoSessao(
                fase=Fase.PRATICA,
                tipo=TipoEvento.ITEM,
                item=ItemReplay.de_item("e1", item),
                resposta_item=RespostaAluno(item_id="e1", escolha="B"),
            ),
        ],
    )


def test_repositorio_de_sessoes_ordena_e_preserva_figuras(tmp_path):
    repo = JsonSessionRepository(tmp_path / "sessions")
    antiga = _registro_falso(finalizada_em=datetime(2026, 1, 1, 11, tzinfo=UTC))
    nova = _registro_falso(finalizada_em=datetime(2026, 1, 2, 11, tzinfo=UTC))
    repo.save(antiga)
    repo.save(nova)

    assert [resumo.id for resumo in repo.list("ana")] == [nova.id, antiga.id]
    carregada = repo.load("ana", nova.id)
    assert carregada.eventos[0].turno_imagens[0].descricao == "Parábola"
    assert carregada.eventos[1].item.imagens[0].descricao == "Parábola"

    with pytest.raises(FileExistsError):
        repo.save(nova)
    with pytest.raises(ValueError):
        repo.load("../fora", nova.id)


def test_ciclo_concluido_salva_sessao_no_historico(kb, banco, executor, tmp_path):
    orq, _ = montar(kb, banco, executor, tmp_path)
    repo = JsonSessionRepository(tmp_path / "sessions")
    sessao = SessaoWeb(orq, "ana", Topico.FUNCOES, sessoes=repo).iniciar()
    while (pedido := esperar_pedido(sessao)) is not None:
        if pedido.turno is not None:
            sessao.learner.responder("acho que é -b/2a")
        else:
            sessao.learner.responder(RespostaAluno(item_id=pedido.item_id, escolha="A"))

    assert sessao.estado == EstadoSessao.CONCLUIDA, sessao.erro
    resumos = repo.list("ana")
    assert len(resumos) == 1
    registro = repo.load("ana", resumos[0].id)
    assert registro.resultado.aluno_id == "ana"
    assert any(evento.tipo == TipoEvento.TUTOR for evento in registro.eventos)
    assert any(evento.tipo == TipoEvento.ITEM for evento in registro.eventos)
    assert any(evento.tipo == TipoEvento.NOTIFICACAO for evento in registro.eventos)


def test_tela_abre_sessao_anterior_em_modo_leitura(app, tmp_path):
    repo = JsonSessionRepository(tmp_path / "sessions")
    registro = _registro_falso()
    repo.save(registro)
    repo.save(_registro_falso(topico=Topico.DINAMICA, finalizada_em=datetime(2026, 1, 3, 11, tzinfo=UTC)))

    app.session_state["aluno"] = "ana"
    app.session_state["sessao_historica"] = {"aluno_id": "ana", "sessao_id": registro.id}
    app.run()

    assert not app.exception
    assert any("Funções" in header.value for header in app.header)
    assert any("Mensagem do tutor" in markdown.value for markdown in app.markdown)
    assert any("Dinâmica" in expander.label for expander in app.expander)
    assert any(button.label == "← Voltar ao início" for button in app.button)


def test_tela_mantem_questao_desabilitada_enquanto_corrige(app):
    item = Exercicio(
        id="e1",
        topico=Topico.FUNCOES,
        enunciado="Quanto é f(2) para f(x) = x²?",
        alternativas=["2", "4", "6", "8", "10"],
        gabarito="B",
    )  # fmt: skip
    sessao = SessaoFalsa(Pedido(fase=Fase.PRATICA, item_id="e1", item=item))
    app.session_state["sessao"] = sessao
    app.run()

    app.radio[0].set_value("A")
    next(button for button in app.button if button.label == "Enviar resposta").click().run()

    assert not app.exception
    assert app.radio[0].disabled
    assert sessao.learner.pedido_em_transicao is not None


def test_tela_nao_mantem_formulario_de_diagnostico_enquanto_agentes_trabalham(app):
    item = Exercicio(
        id="d1",
        topico=Topico.FUNCOES,
        enunciado="Quanto é f(2) para f(x) = x²?",
        alternativas=["2", "4", "6", "8", "10"],
        gabarito="B",
    )  # fmt: skip
    sessao = SessaoFalsa(Pedido(fase=Fase.DIAGNOSTICO, item_id="d1", item=item))
    app.session_state["sessao"] = sessao
    app.run()

    app.radio[0].set_value("A")
    next(button for button in app.button if button.label == "Enviar resposta").click().run()

    assert not app.exception
    assert not app.radio
    assert sessao.learner.pedido_em_transicao is not None
