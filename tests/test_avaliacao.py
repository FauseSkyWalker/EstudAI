import pytest
from fakes import FakeChatModel
from test_orquestrador import handlers, montar

from estudai.application.orquestrador import ConfigSessao
from estudai.domain.curriculum import Topico
from estudai.domain.perfil import NivelDominio
from estudai.domain.questoes import Particao
from estudai.evaluation.aluno_simulado import AlunoSimulado
from estudai.evaluation.estatistica import bootstrap_ic, cohen_kappa, fleiss_kappa, wilcoxon_pareado
from estudai.evaluation.experimento import Condicao, Experimento
from estudai.evaluation.personas import PERSONAS, Persona
from estudai.evaluation.relatorio import gerar_relatorio
from estudai.infrastructure.persistence.json_profiles import JsonProfileRepository
from estudai.infrastructure.sandbox.python_executor import SubprocessPythonExecutor

T = Topico.FUNCOES


def test_dina_usa_numeros_aleatorios_comuns(banco):
    persona = Persona(id="p", descricao="", dominados={T: []}, slip=0.0, guess=0.0)
    aluno = AlunoSimulado(persona, T, seed=1)
    bateria = banco.find(T, Particao.BATERIA)
    assert aluno.responder_bateria(bateria) == [False] * len(bateria)  # guess=0: erra tudo

    aluno.aprender(["funcao_afim", "atrito"])  # conceito de outro tópico é ignorado
    assert aluno.dominados == {"funcao_afim"}
    respostas = aluno.responder_bateria(bateria)
    acertos = [q.conceitos for q, ok in zip(bateria, respostas, strict=True) if ok]
    assert acertos == [["funcao_afim"]]  # só muda o item cujo conceito foi aprendido


def test_dina_e_deterministico_por_seed(banco):
    persona = PERSONAS[1]
    bateria = banco.find(T, Particao.BATERIA)
    a = AlunoSimulado(persona, T, seed=3).responder_bateria(bateria)
    assert a == AlunoSimulado(persona, T, seed=3).responder_bateria(bateria)


def test_personas_tem_niveis_verdadeiros_variados():
    niveis = {p.nivel_verdadeiro(t) for p in PERSONAS for t in Topico}
    assert niveis == set(NivelDominio)


def test_estatisticas():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    assert fleiss_kappa([["A", "A", "A"], ["B", "B", "B"]]) == 1.0
    assert wilcoxon_pareado([1, 2, 3], [1, 2, 3])["p_valor"] == 1.0
    assert wilcoxon_pareado([5, 6, 7, 8, 9, 10], [0, 0, 0, 0, 0, 0])["p_valor"] < 0.05
    baixo, alto = bootstrap_ic([1.0, 2.0, 3.0, 4.0])
    assert 1.0 <= baixo < 2.5 < alto <= 4.0


def test_experimento_ponta_a_ponta_e_relatorio(kb, banco, tmp_path):
    executor = SubprocessPythonExecutor(timeout_s=30)
    juiz = FakeChatModel(
        {
            ("juiz", "VeredictoTransicao"): [
                {"explicou_explicitamente": True, "validou_entendimento": True, "justificativa": ""}
            ],
            ("juiz", "NotasTutoria"): [
                {
                    "fidelidade_material": 4,
                    "personalizacao": 4,
                    "aplicabilidade": 3,
                    "profundidade_logica": 4,
                    "justificativa": "",
                }
            ],
            ("juiz", "AvaliacaoItem"): [
                {
                    "vinculo_topico": True,
                    "clareza": 4,
                    "plausibilidade_distratores": 3,
                    "sem_pistas_gramaticais": True,
                    "defeito_critico": False,
                    "justificativa": "",
                }
            ],
        }
    )
    exp = Experimento(
        fabrica_orquestrador=lambda cfg, fonte, perfis: montar(kb, banco, executor, tmp_path, config=cfg, fonte=fonte)[
            0
        ],
        fabrica_perfis=lambda aluno_id: JsonProfileRepository(tmp_path / "perfis"),
        banco=banco,
        llm_aluno=None,
        juizes=[juiz, juiz],
        config_base=ConfigSessao(exercicios_por_sessao=2),
    )
    saida = tmp_path / "resultados.jsonl"
    resultados = exp.executar(list(Condicao), [PERSONAS[0]], [T], saida)

    assert [r.erro for r in resultados] == [None, None, None]
    c1 = next(r for r in resultados if r.condicao == Condicao.C1)
    # as lacunas ensinadas e validadas viram conceitos dominados antes do pós-teste
    assert c1.ciclos[0].conceitos_aprendidos == ["funcao_quadratica", "funcao_exponencial"]
    assert c1.ciclos[0].nivel_estimado == NivelDominio.INTERMEDIARIO

    # retomada: rodar de novo não reexecuta nada
    assert len(exp.executar(list(Condicao), [PERSONAS[0]], [T], saida)) == 3
    assert len(saida.read_text(encoding="utf-8").splitlines()) == 3

    rel = gerar_relatorio(resultados)
    assert {c.condicao for c in rel.condicoes} == set(Condicao)
    assert rel.kappa_juizes_rubrica == 1.0
    assert rel.criterios["4_qualidade_tutoria"] is True
    assert {(c.metrica, c.alvo) for c in rel.comparacoes} == {
        ("ganho", Condicao.C2),
        ("rubrica", Condicao.C2),
        ("ganho", Condicao.C3),
        ("rubrica", Condicao.C3),
    }
    _ = handlers  # reaproveitado via `montar`
