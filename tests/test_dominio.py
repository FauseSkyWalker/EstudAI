import pytest
from pydantic import ValidationError

from estudai.domain.curriculum import Topico
from estudai.domain.perfil import Diagnostico, NivelDominio, PerfilAluno, RegistroErro, TipoErro
from estudai.domain.planejamento import ObjetivoSessao, planejar_pratica
from estudai.domain.questoes import Dificuldade, Exercicio, ItemObjetivo

T = Topico.FUNCOES


def test_perfil_evolui_em_nova_versao_sem_mutar_a_anterior():
    v0 = PerfilAluno.novo("ana")
    diag = Diagnostico(
        topico=T,
        nivel=NivelDominio.INICIANTE,
        conceitos_dominados=["funcao_afim"],
        lacunas=["funcao_quadratica"],
        confianca=0.7,
    )
    erro = RegistroErro(topico=T, item_id="x", tipo=TipoErro.CONCEITUAL, descricao="sinal")
    v1 = v0.evoluir("diag", diagnostico=diag, novos_erros=(erro,), acertos=1, respondidos=3)

    assert (v0.versao, v1.versao) == (0, 1)
    assert not v0.diagnosticado(T) and v1.diagnosticado(T)
    assert v1.estado(T).lacunas == ("funcao_quadratica",)
    assert v1.estado(T).taxa_acerto == pytest.approx(1 / 3)
    assert v1.visao(T)["erros_recentes"][0]["descricao"] == "sinal"


def test_diagnostico_descarta_conceitos_inventados_e_lacuna_tem_precedencia():
    diag = Diagnostico(
        topico=T,
        nivel=NivelDominio.AVANCADO,
        conceitos_dominados=["funcao_afim", "inventado", "funcao_quadratica"],
        lacunas=["funcao_quadratica", "queda_livre"],
    )
    assert diag.conceitos_dominados == ["funcao_afim"]
    assert diag.lacunas == ["funcao_quadratica"]


def test_nivel_por_fracao():
    assert NivelDominio.da_fracao(0.0) == NivelDominio.INICIANTE
    assert NivelDominio.da_fracao(0.5) == NivelDominio.INTERMEDIARIO
    assert NivelDominio.da_fracao(1.0) == NivelDominio.AVANCADO


def test_objetivo_prioriza_lacunas_e_pratica_faz_rodizio():
    perfil = PerfilAluno.novo("ana").evoluir(
        "diag",
        diagnostico=Diagnostico(
            topico=T,
            nivel=NivelDominio.INICIANTE,
            conceitos_dominados=["dominio_imagem"],
            lacunas=["funcao_exponencial"],
        ),
    )
    objetivo = ObjetivoSessao.derivar(perfil, T, max_conceitos=2)
    assert objetivo.conceitos_alvo == ["funcao_exponencial", "funcao_afim"]

    plano = planejar_pratica(perfil, T, n=4, foco=2)
    assert [p.conceito for p in plano] == ["funcao_exponencial", "funcao_afim"] * 2
    assert {p.dificuldade for p in plano} == {Dificuldade.FACIL}


def test_item_aceita_4_ou_5_alternativas_e_valida_gabarito():
    ItemObjetivo(enunciado="?", alternativas=["a", "b", "c", "d"], gabarito="d")  # Unicamp: 4 alternativas
    with pytest.raises(ValidationError):
        ItemObjetivo(enunciado="?", alternativas=["a", "b", "c", "d"], gabarito="E")
    with pytest.raises(ValidationError):
        ItemObjetivo(enunciado="?", alternativas=["a", "b", "c"], gabarito="A")


def test_exercicio_filtra_conceitos_de_outro_topico():
    ex = Exercicio(
        id="e", topico=T, conceitos=["funcao_afim", "atrito"], enunciado="?", alternativas=list("abcde"), gabarito="A"
    )
    assert ex.conceitos == ["funcao_afim"]
