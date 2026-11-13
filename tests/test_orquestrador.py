"""Pipeline completo com LLM roteirizado e implementações reais de RAG, sandbox e persistência."""

import re

import pytest
from fakes import FakeChatModel, ScriptedLearner, structured_final, tool_call

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
from estudai.infrastructure.llm.callbacks import UsageLedgerCallback
from estudai.infrastructure.persistence.json_profiles import JsonProfileRepository
from estudai.infrastructure.sandbox.python_executor import SubprocessPythonExecutor

T = Topico.FUNCOES


def _letra_da(request, texto: str) -> str:
    """Letra da alternativa cujo texto é `texto` no enunciado que o resolutor recebeu."""
    prompt = next(m.content for m in request.messages if m.type == "human")
    return re.search(rf"^([A-E])\) {re.escape(texto)}$", prompt, re.MULTILINE).group(1)


def handlers(corretor_suspeita: bool = False, voto_painel: str = "5", figura: dict | None = None) -> dict:
    """`voto_painel` é o TEXTO da alternativa em que o painel vota: as alternativas geradas são
    embaralhadas, então o painel (como o real) resolve pelo conteúdo, não por uma letra fixa."""

    def tutor(request):
        # 1ª rodada: ensina; 2ª: valida os conceitos-alvo (lidos do objetivo no prompt) e encerra
        prompt = next(m.content for m in request.messages if m.type == "human")  # [0] é o SystemMessage
        primeira = "(início da sessão)" in prompt
        alvo = ["funcao_quadratica", "funcao_exponencial"]
        return {
            "conceito_alvo": alvo[0],
            "explicacao": "O vértice fica em x = -b/2a [wb-quad].",
            "pergunta_verificacao": "Onde fica o vértice de x² - 4x?",
            "conceitos_validados": [] if primeira else alvo,
            "objetivo_atingido": not primeira,
            "figura": figura if primeira else None,
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
            tool_call("busca_hibrida", {"consulta": "função quadrática vértice"}, "call-search"),
            structured_final("SelecaoConteudo", {"trechos": ["wb-quad"], "questoes_referencia": []}, "call-final"),
        ],
        "tutor": tutor,
        ("resumo", "ResumoEstudo"): [{"titulo": "Função quadrática", "pontos_chave": ["vértice em -b/2a"]}],
        ("resumo", "RelatorioPratica"): [{"desempenho": "Bom progresso."}],
        "gerador_exercicios": [
            structured_final(
                "RascunhoExercicio",
                {
                    "enunciado": "Quanto vale f(2) se f(x) = x² + 1?",
                    "alternativas": ["4", "5", "6", "7", "8"],
                    "gabarito": "B",
                    "resolucao": "2² + 1 = 5",
                    "codigo_verificacao": "print(2**2 + 1)",
                },
                "call-final",
            )
        ],
        "painel_resolutores": lambda request: {"raciocinio": "", "resposta": _letra_da(request, voto_painel)},
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
    voto_revalidacao="5",
    figuras=None,
    **kw,
):
    config = config or ConfigSessao(exercicios_por_sessao=3)
    ledger = UsageLedger()
    llm = FakeChatModel(handlers(**kw), callbacks=[UsageLedgerCallback(ledger)])

    def validador(voto: str) -> ValidadorExercicios:
        painel_llm = FakeChatModel(handlers(voto_painel=voto), callbacks=[UsageLedgerCallback(ledger)])
        return ValidadorExercicios(
            PainelResolutores([Resolutor(painel_llm) for _ in range(3)]), VerificadorSimbolico(executor)
        )

    conteudo = AgenteConteudo(llm, kb)
    fornecedor = (
        BankExerciseProvider(banco)
        if fonte == "banco"
        else GeneratedExerciseProvider(GeradorExercicios(llm, conteudo, executor), validador("5"))
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
        figuras=figuras,
    )
    return orq, perfis


def test_ciclo_adaptativo_completo(kb, banco, executor, tmp_path):
    orq, perfis = montar(kb, banco, executor, tmp_path)
    aluno = ScriptedLearner(padrao="errada")  # erra tudo
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
    # perfil: v0 criação, v1 diagnóstico inicial, v2 prática (com erros), v3 memória episódica
    historico = perfis.history("ana")
    assert [p.motivo for p in historico] == ["criacao", "diagnostico_inicial", "pratica", "memoria_episodica"]
    assert r.versao_perfil_final == 3 and len(historico[-1].historico_erros) == 3
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


def test_perfil_ja_diagnosticado_nao_repete_diagnostico(kb, banco, executor, tmp_path):
    orq, _ = montar(kb, banco, executor, tmp_path)
    primeiro = orq.executar_ciclo("eva", T, ScriptedLearner(padrao="errada"))
    assert primeiro.diagnostico_inicial is not None

    segundo = orq.executar_ciclo("eva", T, ScriptedLearner(padrao="errada"))

    assert segundo.diagnostico_inicial is None
    assert segundo.sondagem == []
    assert segundo.versao_perfil_inicial == primeiro.versao_perfil_final


def test_perfil_congelado_nao_diagnostica_nem_persiste(kb, banco, executor, tmp_path):
    orq, perfis = montar(kb, banco, executor, tmp_path, config=ConfigSessao(adaptativo=False, exercicios_por_sessao=2))
    r = orq.executar_ciclo("bia", T, ScriptedLearner())
    assert r.diagnostico_inicial is None and r.diagnostico_final is None
    assert r.sondagem == [] and r.versao_perfil_final == 0
    assert perfis.history("bia") == []
    assert len(r.correcoes) == 2  # a prática acontece normalmente


def test_memoria_episodica_chega_ao_tutor_no_ciclo_seguinte(kb, banco, executor, tmp_path):
    orq, perfis = montar(kb, banco, executor, tmp_path)
    orq.executar_ciclo("ana", T, ScriptedLearner(padrao="errada"))  # erra tudo

    episodio = perfis.load("ana").episodios[-1]
    assert episodio.pontos_chave == ("vértice em -b/2a",)  # do Agente de Resumo
    assert episodio.conceitos_com_erro == ("funcao_quadratica",)  # das correções
    assert (episodio.acertos, episodio.respondidos, episodio.objetivo_atingido) == (0, 3, True)

    modelo = orq._tutores[Disciplina.MATEMATICA]._model
    ja_vistas = len(modelo.requests)
    r2 = orq.executar_ciclo("ana", T, ScriptedLearner(padrao="certa"))

    prompts_tutor = [
        m.content
        for req in modelo.requests[ja_vistas:]
        if req.caller == "tutor"
        for m in req.messages
        if m.type == "human"
    ]
    assert prompts_tutor, "o tutor não foi chamado no 2º ciclo"
    assert '"sessoes_anteriores"' in prompts_tutor[0] and "vértice em -b/2a" in prompts_tutor[0]
    # ciclo 2: sem novo diagnóstico; prática + memória = 2 versões novas
    assert r2.versao_perfil_final == r2.versao_perfil_inicial + 2
    assert len(perfis.load("ana").episodios) == 2


def test_perfil_congelado_nao_grava_memoria(kb, banco, executor, tmp_path):
    orq, perfis = montar(kb, banco, executor, tmp_path, config=ConfigSessao(adaptativo=False, exercicios_por_sessao=1))
    orq.executar_ciclo("caio", T, ScriptedLearner())
    orq.executar_ciclo("caio", T, ScriptedLearner())
    assert perfis.history("caio") == []  # C2: nem perfil, nem memória


def test_condicao_banco_usa_questoes_reais_sem_repetir_a_sondagem(kb, banco, executor, tmp_path):
    orq, _ = montar(kb, banco, executor, tmp_path, fonte="banco")
    r = orq.executar_ciclo("caio", T, ScriptedLearner())
    ids = [e.id for e in r.fornecimento.exercicios]
    assert len(ids) == 3 and all(e.origem == OrigemItem.BANCO for e in r.fornecimento.exercicios)
    assert not set(ids) & set(r.sondagem)
    assert all(banco.get(i).particao == Particao.BANCO for i in ids)


def test_gabarito_suspeito_sem_consenso_remove_item_e_notifica(kb, banco, executor, tmp_path):
    # o painel aprova na geração, mas, quando o Corretor suspeita, a revalidação discorda
    orq, _ = montar(kb, banco, executor, tmp_path, corretor_suspeita=True, voto_revalidacao="4")
    aluno = ScriptedLearner(padrao="errada")
    r = orq.executar_ciclo("davi", T, aluno)
    assert len(r.removidos) == 3 and r.correcoes == []
    assert any("removido" in n for n in aluno.notificacoes)
    assert any(m.tipo == TipoMensagem.NOTIFICACAO for m in r.mensagens)
