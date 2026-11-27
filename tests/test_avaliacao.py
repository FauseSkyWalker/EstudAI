import pytest
from fakes import FakeChatModel
from test_orquestrador import handlers, montar

from estudai.application.orquestrador import ConfigSessao
from estudai.domain.curriculum import Topico
from estudai.domain.perfil import NivelDominio
from estudai.domain.questoes import Particao
from estudai.evaluation.aluno_simulado import AlunoSimulado
from estudai.evaluation.estatistica import bootstrap_ic, cohen_kappa, fleiss_kappa, holm, wilcoxon_pareado
from estudai.evaluation.experimento import Experimento
from estudai.evaluation.personas import PERSONAS, Persona
from estudai.evaluation.registros import Condicao
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
    # Holm: monotônico, truncado em 1 e de acordo com o exemplo canônico (m = 4)
    assert holm([0.01, 0.02, 0.03, 0.04]) == pytest.approx([0.04, 0.06, 0.06, 0.06])
    assert holm([0.3, 0.02]) == pytest.approx([0.3, 0.04])
    assert holm([0.9, 0.9]) == pytest.approx([1.0, 1.0])  # truncado em 1
    assert holm([0.2]) == pytest.approx([0.2])
    assert holm([]) == []


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
        fabrica_orquestrador=lambda cfg, fonte, perfis, on_tutoring=None: montar(
            kb, banco, executor, tmp_path, config=cfg, fonte=fonte, on_tutoring=on_tutoring
        )[0],
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
    # as lacunas ensinadas e validadas viram conceitos dominados antes da prática e do pós-teste
    assert c1.ciclos[0].conceitos_aprendidos == ["funcao_quadratica", "funcao_exponencial"]
    # a transição é julgada por todos os juízes; a decisão é do primário e o restante audita
    transicao = c1.ciclos[0].transicoes[0]
    assert len(transicao.avaliacoes) == 2
    assert all(v is not None and v.aprendeu for v in transicao.avaliacoes)
    assert transicao.aprendido is True
    assert c1.ciclos[0].nivel_verdadeiro_inicial == NivelDominio.INICIANTE
    assert c1.ciclos[0].nivel_verdadeiro_final == NivelDominio.INTERMEDIARIO
    assert c1.ciclos[0].nivel_estimado_final == NivelDominio.INTERMEDIARIO
    # snapshot do item inclui a resolução e o código de verificação do gerador
    assert c1.ciclos[0].itens[0].resolucao == "2² + 1 = 5"
    assert c1.ciclos[0].itens[0].codigo_verificacao == "print(2**2 + 1)"
    # custo por agente do sistema é persistido no ciclo
    assert "gerador_exercicios" in c1.ciclos[0].uso_por_agente

    # auditoria: as mensagens do protocolo de cada ciclo ficam num JSONL ao lado dos resultados
    from estudai.domain.mensagens import TipoMensagem
    from estudai.evaluation.artefatos import carregar_mensagens

    registro = c1.ciclos[0]
    assert registro.arquivo_mensagens.startswith("resultados_mensagens/")
    mensagens = carregar_mensagens(saida, registro)
    assert len(mensagens) == registro.mensagens > 0
    assert {TipoMensagem.TUTORIA, TipoMensagem.EXERCICIO, TipoMensagem.CORRECAO} <= {m.tipo for m in mensagens}
    assert registro.arquivo_usage.startswith("resultados_usage/")
    assert (saida.parent / registro.arquivo_usage).exists()

    # retomada: rodar de novo não reexecuta nada
    assert len(exp.executar(list(Condicao), [PERSONAS[0]], [T], saida)) == 3
    assert len(saida.read_text(encoding="utf-8").splitlines()) == 3

    rel = gerar_relatorio(resultados)
    assert {c.condicao for c in rel.condicoes} == set(Condicao)
    assert rel.kappa_juizes_rubrica == 1.0
    assert rel.kappa_transicoes == 1.0
    assert rel.criterios["4_qualidade_tutoria"] is True
    assert rel.comparacoes_ajustadas == 4
    assert {(c.metrica, c.alvo) for c in rel.comparacoes} == {
        ("ganho", Condicao.C2),
        ("rubrica", Condicao.C2),
        ("ganho", Condicao.C3),
        ("rubrica", Condicao.C3),
    }
    # métricas derivadas são calculadas, não armazenadas em dobro
    c1_resumo = next(c for c in rel.condicoes if c.condicao == Condicao.C1)
    assert c1_resumo.taxa_descarte is not None
    assert c1_resumo.taxa_aprovacao_filtro == pytest.approx(1 - c1_resumo.taxa_descarte)
    assert c1_resumo.itens_com_defeito_critico is not None
    assert c1_resumo.validacao_auditada == pytest.approx(1 - c1_resumo.itens_com_defeito_critico)

    # exportação: JSON completo, CSV por execução e Markdown legível
    from estudai.evaluation.exportacao import exportar

    prefixo = tmp_path / "relatorio"
    exportar(rel, resultados, prefixo)
    assert {p.suffix for p in tmp_path.glob("relatorio.*")} == {".json", ".csv", ".md"}
    assert "p_Holm=" in prefixo.with_suffix(".md").read_text(encoding="utf-8")
    assert "condicao,persona_id" in prefixo.with_suffix(".csv").read_text(encoding="utf-8")

    _ = handlers  # reaproveitado via `montar`


def test_snapshot_de_item_descartado_preserva_enunciado_resolucao_e_codigo():
    from estudai.application.pratica import ItemDescartado
    from estudai.domain.questoes import Exercicio
    from estudai.evaluation.registros import registro_descartado

    exercicio = Exercicio(
        id="gen-descartado",
        topico=T,
        conceitos=["funcao_afim"],
        enunciado="Quanto vale f(2)?",
        alternativas=["1", "2", "3", "4", "5"],
        gabarito="B",
        resolucao="f(2) = 2",
        codigo_verificacao="print(2)",
    )
    registro = registro_descartado(ItemDescartado(exercicio=exercicio, motivo="sem consenso"))

    assert registro.id == "gen-descartado"
    assert registro.enunciado == "Quanto vale f(2)?"
    assert registro.resolucao == "f(2) = 2"
    assert registro.codigo_verificacao == "print(2)"
    assert registro.motivo == "sem consenso"


def test_concordancia_entre_juizes_tolera_falhas_e_mede_proximidade():
    from estudai.evaluation.relatorio import _concordancia_adjacente, _kappa

    # notas de 2 juízes por (sessão, dimensão); o 3º item teve falha de um juiz e é ignorado
    notas = [[5, 3], [4, 4], [3], [2, 3], [4, 4]]
    assert _concordancia_adjacente(notas) == pytest.approx(3 / 4)  # só [5, 3] difere em mais de 1 ponto
    k = _kappa(notas)
    assert k is not None and k < 1.0
    assert _kappa([[True, True], [False, False], [True, True]]) == 1.0
    assert _kappa([[4], [3]]) is None  # um juiz só: sem κ
