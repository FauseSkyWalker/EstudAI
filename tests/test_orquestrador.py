"""Pipeline completo com LLM roteirizado e implementações reais de RAG, sandbox e persistência."""

import pytest
from fakes import ScriptedLearner, ScriptedLLM, react_final

from estudai.agents.conteudo import AgenteConteudo
from estudai.agents.corretor import Corretor
from estudai.agents.diagnosticador import Diagnosticador
from estudai.agents.gerador import GeradorExercicios
from estudai.agents.resolutor import Resolutor
from estudai.agents.resumo import AgenteResumo
from estudai.agents.tutor import Tutor
from estudai.application.orquestrador import ConfigSessao, Orquestrador
from estudai.application.pratica import BankExerciseProvider, GeneratedExerciseProvider
from estudai.application.telemetria import UsageLedger
from estudai.application.validacao import PainelResolutores, ValidadorExercicios, VerificadorSimbolico
from estudai.domain.curriculum import Disciplina, Topico
from estudai.domain.mensagens import TipoMensagem
from estudai.domain.questoes import OrigemItem, Particao
from estudai.domain.tutoria import Encerramento
from estudai.infrastructure.llm.decorators import TrackingLLMClient
from estudai.infrastructure.persistence.json_profiles import JsonProfileRepository
from estudai.infrastructure.sandbox.python_executor import SubprocessPythonExecutor

T = Topico.FUNCOES


def handlers(corretor_suspeita: bool = False, voto_painel: str = "B") -> dict:
    def tutor(request):
        # 1ª rodada: ensina; 2ª: valida os conceitos-alvo (lidos do objetivo no prompt) e encerra
        prompt = request.messages[0].content
        primeira = "(início da sessão)" in prompt
        alvo = ["funcao_quadratica", "funcao_exponencial"]
        return {
            "conceito_alvo": alvo[0],
            "explicacao": "O vértice fica em x = -b/2a [wb-quad].",
            "pergunta_verificacao": "Onde fica o vértice de x² - 4x?",
            "conceitos_validados": [] if primeira else alvo,
            "objetivo_atingido": not primeira,
        }

    return {
        ("diagnosticador", "Diagnostico"): [
            {
                "topico": "funcoes",
                "nivel": "intermediario",
                "confianca": 0.6,
                "justificativa": "",
                "conceitos_dominados": ["dominio_imagem", "funcao_afim"],
                "lacunas": ["funcao_quadratica", "funcao_exponencial"],
            }
        ],
        "conteudo": [
            {"pensamento": "", "acao": "busca_hibrida", "entrada": {"consulta": "função quadrática vértice"}},
            react_final({"trechos": ["wb-quad"], "questoes_referencia": []}),
        ],
        "tutor": tutor,
        ("resumo", "ResumoEstudo"): [{"titulo": "Função quadrática", "pontos_chave": ["vértice em -b/2a"]}],
        ("resumo", "RelatorioPratica"): [{"desempenho": "Bom progresso."}],
        "gerador_exercicios": [
            react_final(
                {
                    "enunciado": "Quanto vale f(2) se f(x) = x² + 1?",
                    "alternativas": ["4", "5", "6", "7", "8"],
                    "gabarito": "B",
                    "resolucao": "2² + 1 = 5",
                    "codigo_verificacao": "print(2**2 + 1)",
                }
            )
        ],
        "painel_resolutores": [{"raciocinio": "", "resposta": voto_painel}],
        "corretor": [
            {
                "nota": 0.3,
                "feedback": "Reveja o quadrado.",
                "gabarito_suspeito": corretor_suspeita,
                "erros": [{"tipo": "numerico", "conceito": "funcao_quadratica", "descricao": "calculou 2*2 errado"}],
            }
        ],
    }


@pytest.fixture(scope="module")
def executor():
    return SubprocessPythonExecutor(timeout_s=30)


def montar(
    kb,
    banco,
    executor,
    tmp_path,
    *,
    config=None,
    fonte="gerado",
    voto_revalidacao="B",
    **kw,
):
    config = config or ConfigSessao(exercicios_por_sessao=3)
    ledger = UsageLedger()
    llm = TrackingLLMClient(ScriptedLLM(handlers(**kw)), ledger)

    def validador(voto: str) -> ValidadorExercicios:
        painel_llm = TrackingLLMClient(ScriptedLLM(handlers(voto_painel=voto)), ledger)
        return ValidadorExercicios(
            PainelResolutores([Resolutor(painel_llm) for _ in range(3)]), VerificadorSimbolico(executor)
        )

    conteudo = AgenteConteudo(llm, kb, min_score=0.01)
    fornecedor = (
        BankExerciseProvider(banco)
        if fonte == "banco"
        else GeneratedExerciseProvider(GeradorExercicios(llm, conteudo, executor), validador("B"))
    )
    perfis = JsonProfileRepository(tmp_path)
    orq = Orquestrador(
        diagnosticador=Diagnosticador(llm, max_perguntas=4),
        tutores={d: Tutor(llm, d) for d in Disciplina},
        conteudo=conteudo,
        resumo=AgenteResumo(llm),
        corretor=Corretor(llm),
        fornecedor=fornecedor,
        validador=validador(voto_revalidacao),  # usado quando o Corretor suspeita do gabarito
        perfis=perfis,
        banco=banco,
        config=config,
        ledger=ledger,
    )
    return orq, perfis


def test_ciclo_adaptativo_completo(kb, banco, executor, tmp_path):
    orq, perfis = montar(kb, banco, executor, tmp_path)
    aluno = ScriptedLearner(padrao="A")  # erra tudo
    r = orq.executar_ciclo("ana", T, aluno)

    # diagnóstico inicial com questões do banco (nunca da bateria)
    assert len(r.sondagem) == 4
    assert all(banco.get(i).particao == Particao.BANCO for i in r.sondagem)
    assert r.diagnostico_inicial.lacunas == ["funcao_quadratica", "funcao_exponencial"]
    # objetivo derivado das lacunas; tutoria encerra quando valida os conceitos
    assert r.objetivo.conceitos_alvo == ["funcao_quadratica", "funcao_exponencial"]
    assert r.transcricao.encerramento == Encerramento.OBJETIVO_ATINGIDO
    assert len(r.transcricao.trocas) == 2 and len(aluno.turnos) == 1
    assert [t.doc_id for t in r.material.trechos][0] == "wb-quad"
    # prática: itens gerados e aprovados (verificação simbólica + painel)
    assert len(r.fornecimento.exercicios) == 3 and r.fornecimento.taxa_descarte == 0
    assert all(not c.correta for c in r.correcoes)
    # perfil: v0 criação, v1 diagnóstico inicial, v2 prática (com erros registrados)
    historico = perfis.history("ana")
    assert [p.motivo for p in historico] == ["criacao", "diagnostico_inicial", "pratica"]
    assert r.versao_perfil_final == 2 and len(historico[-1].historico_erros) == 3
    assert r.resumo and r.relatorio
    # protocolo e telemetria
    tipos = {m.tipo for m in r.mensagens}
    assert {
        TipoMensagem.DIAGNOSTICO,
        TipoMensagem.MATERIAL,
        TipoMensagem.TUTORIA,
        TipoMensagem.EXERCICIO,
        TipoMensagem.CORRECAO,
        TipoMensagem.PERFIL,
        TipoMensagem.RESUMO,
    } <= tipos
    assert {
        "diagnosticador",
        "tutor",
        "conteudo",
        "gerador_exercicios",
        "painel_resolutores",
        "corretor",
        "resumo",
    } <= set(r.uso.por_agente)


def test_perfil_congelado_nao_diagnostica_nem_persiste(kb, banco, executor, tmp_path):
    orq, perfis = montar(kb, banco, executor, tmp_path, config=ConfigSessao(adaptativo=False, exercicios_por_sessao=2))
    r = orq.executar_ciclo("bia", T, ScriptedLearner())
    assert r.diagnostico_inicial is None and r.diagnostico_final is None
    assert r.sondagem == [] and r.versao_perfil_final == 0
    assert perfis.history("bia") == []
    assert len(r.correcoes) == 2  # a prática acontece normalmente


def test_condicao_banco_usa_questoes_reais_sem_repetir_a_sondagem(kb, banco, executor, tmp_path):
    orq, _ = montar(kb, banco, executor, tmp_path, fonte="banco")
    r = orq.executar_ciclo("caio", T, ScriptedLearner())
    ids = [e.id for e in r.fornecimento.exercicios]
    assert len(ids) == 3 and all(e.origem == OrigemItem.BANCO for e in r.fornecimento.exercicios)
    assert not set(ids) & set(r.sondagem)
    assert all(banco.get(i).particao == Particao.BANCO for i in ids)


def test_gabarito_suspeito_sem_consenso_remove_item_e_notifica(kb, banco, executor, tmp_path):
    # o painel aprova na geração, mas, quando o Corretor suspeita, a revalidação discorda
    orq, _ = montar(kb, banco, executor, tmp_path, corretor_suspeita=True, voto_revalidacao="C")
    aluno = ScriptedLearner(padrao="A")
    r = orq.executar_ciclo("davi", T, aluno)
    assert len(r.removidos) == 3 and r.correcoes == []
    assert any("removido" in n for n in aluno.notificacoes)
    assert any(m.tipo == TipoMensagem.NOTIFICACAO for m in r.mensagens)
