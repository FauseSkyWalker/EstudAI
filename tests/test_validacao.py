import pytest
from fakes import ScriptedLLM

from estudai.agents.resolutor import Resolutor
from estudai.application.validacao import (
    PainelResolutores,
    StatusVerificacao,
    ValidadorExercicios,
    VerificadorSimbolico,
    _para_float,
    valor_da_alternativa,
)
from estudai.domain.curriculum import Topico
from estudai.domain.questoes import Exercicio
from estudai.infrastructure.sandbox.python_executor import SubprocessPythonExecutor


def exercicio(gabarito="B", codigo="print(2 + 3)", alternativas=("4", "5", "6", "7", "8")) -> Exercicio:
    return Exercicio(
        id="g1",
        topico=Topico.FUNCOES,
        enunciado="Quanto é 2 + 3?",
        alternativas=list(alternativas),
        gabarito=gabarito,
        codigo_verificacao=codigo,
    )


@pytest.fixture(scope="module")
def verificador():
    return VerificadorSimbolico(SubprocessPythonExecutor(timeout_s=30))


def test_valores_de_alternativas():
    assert valor_da_alternativa("12 m/s") == 12
    assert valor_da_alternativa("0,25") == 0.25
    assert valor_da_alternativa("3/4") == 0.75
    assert valor_da_alternativa("entre 2 e 3") is None  # ambíguo


def test_para_float_rejeita_codigo_arbitrario():
    assert _para_float("sqrt(2)") == pytest.approx(1.41421356)
    assert _para_float("__import__('os').system('echo x')") is None
    assert _para_float("open('x')") is None


@pytest.mark.parametrize(
    ("item", "status"),
    [
        (exercicio(), StatusVerificacao.CONFIRMADA),
        (exercicio(gabarito="C"), StatusVerificacao.DIVERGENTE),
        (exercicio(codigo="print(99)"), StatusVerificacao.DIVERGENTE),  # nenhuma alternativa bate
        (exercicio(alternativas=("4", "5", "5,0", "7", "8")), StatusVerificacao.DIVERGENTE),  # duas corretas
        (exercicio(codigo=None), StatusVerificacao.INCONCLUSIVA),
        (exercicio(alternativas=("quatro", "cinco", "seis", "sete", "oito")), StatusVerificacao.INCONCLUSIVA),
    ],
)
def test_verificacao_simbolica(verificador, item, status):
    assert verificador.verificar(item).status == status


def painel(*respostas: str) -> PainelResolutores:
    return PainelResolutores(
        [Resolutor(ScriptedLLM({"painel_resolutores": [{"raciocinio": "", "resposta": r}]})) for r in respostas]
    )


def test_aprova_com_maioria_quando_verificacao_confirma(verificador):
    resultado = ValidadorExercicios(painel("B", "B", "C"), verificador).validar(exercicio())
    assert resultado.aprovado and resultado.concordancia == pytest.approx(2 / 3)


def test_exige_unanimidade_sem_verificacao(verificador):
    item = exercicio(codigo=None)
    assert not ValidadorExercicios(painel("B", "B", "C"), verificador).validar(item).aprovado
    assert ValidadorExercicios(painel("B", "B", "B"), verificador).validar(item).aprovado


def test_divergencia_simbolica_descarta_mesmo_com_consenso(verificador):
    resultado = ValidadorExercicios(painel("C", "C", "C"), verificador).validar(exercicio(gabarito="C"))
    assert not resultado.aprovado and "divergente" in resultado.motivo


def test_voto_perdido_conta_como_discordancia(verificador):
    falho = Resolutor(ScriptedLLM({"painel_resolutores": ["lixo"]}), max_repairs=0)
    p = PainelResolutores([*painel("B", "B")._resolutores, falho])
    resultado = ValidadorExercicios(p, verificador).validar(exercicio(codigo=None))
    assert len(resultado.votos) == 2 and not resultado.aprovado
