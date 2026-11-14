"""Figuras desenhadas pelos agentes: renderização na sandbox, schema do Tutor e integração no ciclo."""

import pytest
from fakes import FIGURA, RendererFalso, ScriptedLearner
from test_orquestrador import T, montar

from estudai.domain.figura import (
    Figura,
    FiguraError,
    FormatoFigura,
    PedidoFigura,
    TipoFigura,
)
from estudai.domain.planejamento import ObjetivoSessao
from estudai.domain.tutoria import TranscricaoTutoria, TrocaTutoria, TurnoTutor
from estudai.infrastructure.figures import CompositeFigureRenderer, MatplotlibRenderer, MermaidRenderer
from estudai.infrastructure.sandbox.python_executor import SubprocessPythonExecutor

PNG = b"\x89PNG\r\n\x1a\n"
PARABOLA = PedidoFigura(
    codigo=(
        "x = np.linspace(-1, 5, 100)\n"
        "plt.plot(x, x**2 - 4*x + 3)\n"
        "plt.title('Parábola $y = x^2 - 4x + 3$')\n"
        "plt.xlabel('x'); plt.ylabel('y')"
    ),
    descricao="Parábola com raízes 1 e 3",
)


@pytest.fixture(scope="module")
def renderer():
    return MatplotlibRenderer()


@pytest.fixture(scope="module")
def executor():
    return SubprocessPythonExecutor(timeout_s=30)


# --- renderização ------------------------------------------------------------------------


def test_desenha_png_valido(renderer):
    figura = renderer.render(PARABOLA)
    assert figura.descricao == PARABOLA.descricao
    assert figura.png.startswith(PNG) and len(figura.png) > 1_000


@pytest.mark.parametrize(
    "codigo",
    [
        "raise ValueError('falhou')",  # erro no código do modelo
        "x = 1",  # não desenhou nada
        "import socket; socket.create_connection(('example.com', 80))",  # rede bloqueada
    ],
)
def test_codigo_ruim_ou_malicioso_vira_figura_error(renderer, codigo):
    with pytest.raises(FiguraError):
        renderer.render(PedidoFigura(codigo=codigo, descricao="x"))


def test_escrita_em_disco_e_bloqueada_inclusive_via_os(renderer, tmp_path):
    arquivo = tmp_path / "invasao.txt"
    pasta = tmp_path / "invasao"
    tentativas = [
        f"open({str(arquivo)!r}, 'w').write('x')",
        f"import os; os.open({str(arquivo)!r}, os.O_WRONLY | os.O_CREAT)",
        f"import os; os.mkdir({str(pasta)!r})",
    ]
    for codigo in tentativas:
        with pytest.raises(FiguraError):
            renderer.render(PedidoFigura(codigo=codigo, descricao="x"))
    assert not arquivo.exists() and not pasta.exists()


# --- mermaid -----------------------------------------------------------------------------


DIAGRAMA = PedidoFigura(
    tipo=TipoFigura.DIAGRAMA,
    codigo="flowchart TD\n  A[Função afim] --> B[Gráfico: reta]",
    descricao="Fluxo da função afim",
)


def test_mermaid_valido_vira_figura_de_diagrama():
    figura = MermaidRenderer().render(DIAGRAMA)
    assert figura.formato == FormatoFigura.MERMAID
    assert figura.conteudo.startswith("flowchart TD")
    assert figura.descricao == DIAGRAMA.descricao


@pytest.mark.parametrize(
    "fonte",
    [
        "A --> B",  # sem tipo de diagrama
        'flowchart TD\n  click A href "https://example.com"',  # link
        "flowchart TD\n  A[<script>alert(1)</script>]",  # HTML
        "flowchart TD\n  A[javascript:alert(1)]",  # javascript:
        "%%{init: {'theme': 'dark'}}%%\nflowchart TD\n  A --> B",  # diretiva
    ],
)
def test_mermaid_invalido_vira_figura_error(fonte):
    with pytest.raises(FiguraError):
        MermaidRenderer().render(PedidoFigura(tipo=TipoFigura.DIAGRAMA, codigo=fonte, descricao="x"))


def test_mermaid_rejeita_pedido_de_grafico():
    with pytest.raises(FiguraError):
        MermaidRenderer().render(PARABOLA)


def test_composite_despacha_por_tipo():
    composite = CompositeFigureRenderer(
        {
            TipoFigura.GRAFICO: RendererFalso(),
            TipoFigura.DIAGRAMA: MermaidRenderer(),
        }
    )
    assert composite.render(PARABOLA).formato == FormatoFigura.PNG
    assert composite.render(DIAGRAMA).formato == FormatoFigura.MERMAID


# --- domínio -----------------------------------------------------------------------------


def test_png_fica_fora_do_schema_do_modelo_e_dos_logs():
    schema = TurnoTutor.model_json_schema()
    assert "figuras" in schema["properties"] and "imagens" not in schema["properties"]

    turno = TurnoTutor(
        conceito_alvo="funcao_quadratica",
        explicacao="Veja o gráfico.",
        pergunta_verificacao="Quais as raízes?",
        figuras=[PARABOLA],
        imagens=[Figura(descricao=PARABOLA.descricao, formato=FormatoFigura.PNG, conteudo="AAAA")],
    )
    assert "imagens" not in turno.model_dump() and "AAAA" not in turno.model_dump_json()


def test_transcricao_menciona_as_figuras():
    turno = TurnoTutor(conceito_alvo="c", explicacao="e", pergunta_verificacao="p?", figuras=[PARABOLA, DIAGRAMA])
    objetivo = ObjetivoSessao(topico=T, conceitos_alvo=["funcao_quadratica"], nivel_aluno=None, descricao="")
    transcricao = TranscricaoTutoria(objetivo=objetivo, trocas=[TrocaTutoria(rodada=1, turno=turno)])
    texto = transcricao.como_texto()
    assert "TUTOR (figura): Parábola com raízes 1 e 3" in texto
    assert "TUTOR (figura): Fluxo da função afim" in texto


# --- integração no ciclo ------------------------------------------------------------------


def test_ciclo_entrega_as_figuras_ao_aluno(kb, banco, executor, tmp_path):
    renderer = RendererFalso()
    orq, _ = montar(kb, banco, executor, tmp_path, renderer=renderer, figuras=[FIGURA])
    aluno = ScriptedLearner(padrao="certa")
    r = orq.executar_ciclo("ana", T, aluno)

    assert [p.descricao for p in renderer.pedidos] == ["Reta identidade"]
    assert aluno.turnos[0].imagens[0].descricao == "Reta identidade"
    assert r.transcricao.trocas[0].turno.figuras[0].descricao == "Reta identidade"


def test_ciclo_entrega_multiplas_figuras(kb, banco, executor, tmp_path):
    segunda = {"tipo": "diagrama", "codigo": "flowchart TD\n  A --> B", "descricao": "Fluxo"}
    renderer = RendererFalso()
    orq, _ = montar(kb, banco, executor, tmp_path, renderer=renderer, figuras=[FIGURA, segunda])
    aluno = ScriptedLearner(padrao="certa")
    r = orq.executar_ciclo("ana", T, aluno)

    assert [p.descricao for p in renderer.pedidos] == ["Reta identidade", "Fluxo"]
    assert [i.descricao for i in aluno.turnos[0].imagens] == ["Reta identidade", "Fluxo"]
    assert [f.descricao for f in r.transcricao.trocas[0].turno.figuras] == ["Reta identidade", "Fluxo"]


def test_ciclo_respeita_o_limite_de_figuras(kb, banco, executor, tmp_path):
    segunda = {"codigo": "plt.plot([1, 0], [0, 1])", "descricao": "Outra reta"}
    renderer = RendererFalso()
    orq, _ = montar(kb, banco, executor, tmp_path, renderer=renderer, figuras=[FIGURA, segunda], max_figuras=1)
    aluno = ScriptedLearner(padrao="certa")
    orq.executar_ciclo("ana", T, aluno)

    assert [p.descricao for p in renderer.pedidos] == ["Reta identidade"]
    assert [i.descricao for i in aluno.turnos[0].imagens] == ["Reta identidade"]


def test_figura_que_falha_nao_descarta_as_outras(kb, banco, executor, tmp_path):
    segunda = {"codigo": "plt.plot([1, 0], [0, 1])", "descricao": "Outra reta"}
    renderer = RendererFalso(falhar_em={"Reta identidade"})
    orq, _ = montar(kb, banco, executor, tmp_path, renderer=renderer, figuras=[FIGURA, segunda])
    aluno = ScriptedLearner(padrao="certa")
    r = orq.executar_ciclo("ana", T, aluno)

    assert [i.descricao for i in aluno.turnos[0].imagens] == ["Outra reta"]
    assert [f.descricao for f in r.transcricao.trocas[0].turno.figuras] == ["Outra reta"]


def test_todas_as_figuras_falhando_nao_derrubam_o_ciclo(kb, banco, executor, tmp_path):
    orq, _ = montar(kb, banco, executor, tmp_path, renderer=RendererFalso(falha=True), figuras=[FIGURA])
    aluno = ScriptedLearner(padrao="certa")
    r = orq.executar_ciclo("ana", T, aluno)

    assert aluno.turnos[0].imagens == []
    assert r.transcricao.trocas[0].turno.figuras == []  # a transcrição não cita figura que o aluno não viu


def test_sem_renderer_os_pedidos_de_figura_sao_ignorados(kb, banco, executor, tmp_path):
    orq, _ = montar(kb, banco, executor, tmp_path, figuras=[FIGURA])
    aluno = ScriptedLearner(padrao="certa")
    orq.executar_ciclo("ana", T, aluno)
    assert aluno.turnos[0].figuras == [] and aluno.turnos[0].imagens == []


def test_terminal_salva_a_figura_e_mostra_o_caminho(renderer, tmp_path):
    from io import StringIO

    from rich.console import Console

    from estudai.interfaces.console import ConsoleLearner

    saida = StringIO()
    learner = ConsoleLearner(Console(file=saida, width=300), pasta_figuras=tmp_path / "figuras")
    figura = renderer.render(PARABOLA)
    learner._mostrar_figura(figura)

    (arquivo,) = (tmp_path / "figuras").glob("*.png")
    assert arquivo.read_bytes() == figura.png
    assert arquivo.name in saida.getvalue()


# --- figuras em questões geradas ----------------------------------------------------------


def test_questao_gerada_recebe_figura_e_legenda(kb, banco, executor, tmp_path):
    orq, _ = montar(kb, banco, executor, tmp_path, renderer=RendererFalso(), figuras_item=[FIGURA])
    r = orq.executar_ciclo("ana", T, ScriptedLearner(padrao="certa"))

    assert r.fornecimento.exercicios
    for exercicio in r.fornecimento.exercicios:
        assert [f.descricao for f in exercicio.figuras] == ["Reta identidade"]
        assert [i.descricao for i in exercicio.imagens] == ["Reta identidade"]
        assert "[Figura: Reta identidade]" in exercicio.formatar()


def test_questao_gerada_respeita_o_limite_de_figuras(kb, banco, executor, tmp_path):
    segunda = {"codigo": "plt.plot([1, 0], [0, 1])", "descricao": "Outra reta"}
    orq, _ = montar(
        kb,
        banco,
        executor,
        tmp_path,
        renderer=RendererFalso(),
        figuras_item=[FIGURA, segunda],
        max_figuras_exercicio=1,
    )
    r = orq.executar_ciclo("ana", T, ScriptedLearner(padrao="certa"))

    for exercicio in r.fornecimento.exercicios:
        assert [f.descricao for f in exercicio.figuras] == ["Reta identidade"]
        assert [i.descricao for i in exercicio.imagens] == ["Reta identidade"]
        assert "Outra reta" not in exercicio.formatar()


def test_falha_de_figura_da_questao_mantem_a_legenda(kb, banco, executor, tmp_path):
    orq, _ = montar(
        kb,
        banco,
        executor,
        tmp_path,
        renderer=RendererFalso(falha=True),
        figuras_item=[FIGURA],
    )
    r = orq.executar_ciclo("ana", T, ScriptedLearner(padrao="certa"))

    assert r.fornecimento.exercicios
    for exercicio in r.fornecimento.exercicios:
        assert [f.descricao for f in exercicio.figuras] == ["Reta identidade"]
        assert exercicio.imagens == []
        assert "[Figura: Reta identidade]" in exercicio.formatar()


def test_sem_renderer_a_questao_nao_cita_figura(kb, banco, executor, tmp_path):
    orq, _ = montar(kb, banco, executor, tmp_path, figuras_item=[FIGURA])
    r = orq.executar_ciclo("ana", T, ScriptedLearner(padrao="certa"))

    assert r.fornecimento.exercicios
    for exercicio in r.fornecimento.exercicios:
        assert exercicio.figuras == [] and exercicio.imagens == []
        assert "[Figura:" not in exercicio.formatar()


def test_terminal_salva_figura_da_questao(renderer, tmp_path, monkeypatch):
    from io import StringIO

    from rich.console import Console

    from estudai.domain.questoes import Exercicio
    from estudai.interfaces.console import ConsoleLearner

    figura = renderer.render(PARABOLA)
    item = Exercicio(
        id="e1",
        topico=T,
        enunciado="Veja a figura.",
        alternativas=list("abcde"),
        gabarito="A",
        descricoes_imagens=[figura.descricao],
        imagens=[figura],
    )
    learner = ConsoleLearner(Console(file=StringIO(), width=300), pasta_figuras=tmp_path / "figuras")
    monkeypatch.setattr("estudai.interfaces.console.Prompt.ask", lambda *a, **k: "A")

    learner.solve("e1", item)

    (arquivo,) = (tmp_path / "figuras").glob("*.png")
    assert arquivo.read_bytes() == figura.png
