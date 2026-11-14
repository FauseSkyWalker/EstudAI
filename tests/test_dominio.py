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


def _episodio_de_teste(correcoes=None, trocas=True):
    from estudai.domain.correcao import Correcao, ErroIdentificado, RelatorioPratica
    from estudai.domain.memoria import registrar_episodio
    from estudai.domain.tutoria import Encerramento, ResumoEstudo, TranscricaoTutoria, TrocaTutoria, TurnoTutor

    objetivo = ObjetivoSessao(topico=T, conceitos_alvo=["funcao_quadratica"], nivel_aluno=None, descricao="")
    transcricao = TranscricaoTutoria(objetivo=objetivo, encerramento=Encerramento.LIMITE_RODADAS)
    if trocas:
        turno = TurnoTutor(conceito_alvo="funcao_quadratica", explicacao="...", pergunta_verificacao="?")
        transcricao.trocas.append(TrocaTutoria(rodada=1, turno=turno, resposta_aluno="r"))
    resumo = ResumoEstudo(
        titulo="t", pontos_chave=["a", "a", "b", "c", "d", "e", "f"], erros_comuns=["sinal"], proximos_passos=["p1"]
    )
    relatorio = RelatorioPratica(desempenho="ok", pontos_a_melhorar=["sinal", "vértice"], recomendacoes=["p1", "p2"])
    if correcoes is None:
        erro = ErroIdentificado(tipo=TipoErro.CONCEITUAL, conceito="funcao_quadratica", descricao="x")
        correcoes = [
            Correcao(item_id="e1", escolha="A", gabarito="B", correta=False, nota=0, feedback="", erros=[erro]),
            Correcao(item_id="e2", escolha="B", gabarito="B", correta=True, nota=1, feedback=""),
        ]
    return registrar_episodio(objetivo, transcricao, resumo, relatorio, correcoes)


def test_episodio_junta_resumo_relatorio_e_correcoes_sem_repetir():
    ep = _episodio_de_teste()
    assert ep.pontos_chave == ("a", "b", "c", "d", "e")  # sem repetição e no máximo 5
    assert ep.erros_comuns == ("sinal", "vértice")
    assert ep.proximos_passos == ("p1", "p2")
    assert ep.conceitos_com_erro == ("funcao_quadratica",)
    assert (ep.acertos, ep.respondidos, ep.objetivo_atingido) == (1, 2, False)
    assert _episodio_de_teste(correcoes=[], trocas=False) is None  # nada aconteceu: não há episódio


def test_visao_traz_so_as_ultimas_sessoes_do_topico():
    perfil = PerfilAluno.novo("ana")
    for i in range(4):
        ep = _episodio_de_teste().model_copy(update={"pontos_chave": (f"sessao {i}",)})
        perfil = perfil.evoluir("memoria_episodica", topico=T, episodio=ep)
    outro = _episodio_de_teste().model_copy(update={"topico": Topico.DINAMICA})
    perfil = perfil.evoluir("memoria_episodica", topico=Topico.DINAMICA, episodio=outro)

    sessoes = perfil.visao(T, max_episodios=3)["sessoes_anteriores"]
    assert [s["pontos_chave"] for s in sessoes] == [["sessao 1"], ["sessao 2"], ["sessao 3"]]
    assert sessoes[0]["acertos"] == "1/2"


def test_perfil_antigo_sem_memoria_continua_carregando():
    antigo = PerfilAluno.novo("ana").model_dump(mode="json")
    antigo.pop("episodios")
    assert PerfilAluno.model_validate(antigo).episodios == ()


def test_embaralhar_alternativas_preserva_a_resposta_correta():
    from collections import Counter

    def item(i):
        return Exercicio(
            id=f"gen-{i}", topico=T, enunciado="?", alternativas=["10", "20", "30", "40", "50"], gabarito="C"
        )

    for i in range(50):
        original, embaralhado = item(i), item(i).com_alternativas_embaralhadas()
        assert sorted(embaralhado.alternativas) == sorted(original.alternativas)
        assert embaralhado.alternativa(embaralhado.gabarito) == "30"  # a resposta certa continua certa
        assert embaralhado == item(i).com_alternativas_embaralhadas()  # reprodutível pelo id
    posicoes = Counter(item(i).com_alternativas_embaralhadas().gabarito for i in range(200))
    assert set(posicoes) == {"A", "B", "C", "D", "E"} and max(posicoes.values()) < 70  # sem posição dominante


def test_formatar_inclui_descricoes_de_imagens_sem_duplicar():
    base = dict(id="e", topico=T, alternativas=list("abcde"), gabarito="A")
    inline = Exercicio(
        **base,
        enunciado="Veja [Figura: gráfico de f].",
        descricoes_imagens=["gráfico de f"],
    )
    assert inline.formatar().count("[Figura: gráfico de f]") == 1

    pendente = Exercicio(**base, enunciado="Veja a figura.", descricoes_imagens=["gráfico de f"])
    assert "[Figura: gráfico de f]" in pendente.formatar()
