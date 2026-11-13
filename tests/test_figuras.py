"""Figuras desenhadas pelos agentes: renderização na sandbox, schema do Tutor e integração no ciclo."""

import pytest
from fakes import ScriptedLearner
from test_orquestrador import T, montar

from estudai.domain.figura import Figura, FiguraError, PedidoFigura
from estudai.domain.planejamento import ObjetivoSessao
from estudai.domain.tutoria import TranscricaoTutoria, TrocaTutoria, TurnoTutor
from estudai.infrastructure.sandbox.figuras import MatplotlibRenderer
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
        "open('invasao.txt', 'w').write('x')",  # gravar arquivo é bloqueado pela sandbox
        "import socket; socket.create_connection(('example.com', 80))",  # rede bloqueada
    ],
)
def test_codigo_ruim_ou_malicioso_vira_figura_error(renderer, codigo, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FiguraError):
        renderer.render(PedidoFigura(codigo=codigo, descricao="x"))
    assert not (tmp_path / "invasao.txt").exists()


# --- domínio -----------------------------------------------------------------------------


def test_png_fica_fora_do_schema_do_modelo_e_dos_logs():
    schema = TurnoTutor.model_json_schema()
    assert "figura" in schema["properties"] and "imagem" not in schema["properties"]

    turno = TurnoTutor(
        conceito_alvo="funcao_quadratica",
        explicacao="Veja o gráfico.",
        pergunta_verificacao="Quais as raízes?",
        figura=PARABOLA,
        imagem=Figura(descricao=PARABOLA.descricao, png_base64="AAAA"),
    )
    assert "imagem" not in turno.model_dump() and "AAAA" not in turno.model_dump_json()


def test_transcricao_menciona_a_figura():
    turno = TurnoTutor(conceito_alvo="c", explicacao="e", pergunta_verificacao="p?", figura=PARABOLA)
    objetivo = ObjetivoSessao(topico=T, conceitos_alvo=["funcao_quadratica"], nivel_aluno=None, descricao="")
    transcricao = TranscricaoTutoria(objetivo=objetivo, trocas=[TrocaTutoria(rodada=1, turno=turno)])
    assert "TUTOR (figura): Parábola com raízes 1 e 3" in transcricao.como_texto()


# --- integração no ciclo ------------------------------------------------------------------


class RendererFalso:
    def __init__(self, falha: bool = False) -> None:
        self.falha, self.pedidos = falha, []

    def render(self, pedido: PedidoFigura) -> Figura:
        self.pedidos.append(pedido)
        if self.falha:
            raise FiguraError("código inválido")
        return Figura(descricao=pedido.descricao, png_base64="iVBORw0KGgo=")


FIGURA = {"codigo": "plt.plot([0, 1], [0, 1])", "descricao": "Reta identidade"}


def test_ciclo_entrega_a_figura_ao_aluno(kb, banco, executor, tmp_path):
    figuras = RendererFalso()
    orq, _ = montar(kb, banco, executor, tmp_path, figuras=figuras, figura=FIGURA)
    aluno = ScriptedLearner(padrao="certa")
    r = orq.executar_ciclo("ana", T, aluno)

    assert [p.descricao for p in figuras.pedidos] == ["Reta identidade"]
    assert aluno.turnos[0].imagem is not None and aluno.turnos[0].imagem.descricao == "Reta identidade"
    assert r.transcricao.trocas[0].turno.figura is not None


def test_figura_que_falha_e_descartada_sem_derrubar_o_ciclo(kb, banco, executor, tmp_path):
    orq, _ = montar(kb, banco, executor, tmp_path, figuras=RendererFalso(falha=True), figura=FIGURA)
    aluno = ScriptedLearner(padrao="certa")
    r = orq.executar_ciclo("ana", T, aluno)

    assert aluno.turnos[0].imagem is None
    assert r.transcricao.trocas[0].turno.figura is None  # a transcrição não cita figura que o aluno não viu


def test_sem_renderer_o_pedido_de_figura_e_ignorado(kb, banco, executor, tmp_path):
    orq, _ = montar(kb, banco, executor, tmp_path, figura=FIGURA)
    aluno = ScriptedLearner(padrao="certa")
    orq.executar_ciclo("ana", T, aluno)
    assert aluno.turnos[0].figura is None and aluno.turnos[0].imagem is None


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
