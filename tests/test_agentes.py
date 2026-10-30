import pytest
from fakes import FakeChatModel, react_final
from pydantic import BaseModel

from estudai.agents.base import Agent, AgentOutputError
from estudai.agents.conteudo import AgenteConteudo
from estudai.agents.corretor import Corretor
from estudai.agents.diagnosticador import Diagnosticador, Evidencia
from estudai.agents.react import ReActRunner, Tool
from estudai.domain.correcao import RespostaAluno
from estudai.domain.curriculum import Topico
from estudai.domain.mensagens import Papel
from estudai.domain.perfil import NivelDominio, PerfilAluno
from estudai.domain.questoes import Dificuldade, Exercicio
from estudai.infrastructure.llm.callbacks import UsageLedgerCallback


class Saida(BaseModel):
    valor: int


class AgenteTeste(Agent):
    papel = Papel.TUTOR

    def pedir(self) -> Saida:
        return self._ask("sys", "prompt", Saida)


def test_saida_invalida_e_reenviada_com_instrucao_de_correcao():
    llm = FakeChatModel({"tutor": ['{"valor": "abc"}', '{"valor": 7}']})
    assert AgenteTeste(llm, UsageLedgerCallback()).pedir().valor == 7
    reparo = llm.requests[1].messages[-1].content
    assert "inválida" in reparo and "valor" in reparo


def test_desiste_apos_limite_de_reparos():
    llm = FakeChatModel({"tutor": ["lixo"]})
    with pytest.raises(AgentOutputError):
        AgenteTeste(llm, UsageLedgerCallback(), max_repairs=1).pedir()
    assert len(llm.requests) == 2


class Args(BaseModel):
    x: int


def test_react_usa_ferramenta_e_recupera_de_erros():
    passos = [
        {"pensamento": "?", "acao": "inexistente", "entrada": {}},
        {"pensamento": "dobrar", "acao": "dobro", "entrada": {"x": "não-número"}},
        {"pensamento": "dobrar", "acao": "dobro", "entrada": {"x": 21}},
        react_final({"valor": 42}),
    ]
    llm = FakeChatModel({"teste": passos})
    dobro = Tool("dobro", "dobra x", Args, lambda a: str(a.x * 2))
    saida, trace = ReActRunner(llm, caller="teste", max_steps=5).run("sys", "tarefa", [dobro], Saida)

    assert saida.valor == 42
    observacoes = [obs for _, obs in trace.steps]
    assert "não existe" in observacoes[0]
    assert "Argumentos inválidos" in observacoes[1]
    assert observacoes[2] == "42"


def test_react_estoura_limite_de_passos():
    llm = FakeChatModel({"teste": [{"pensamento": "", "acao": "dobro", "entrada": {"x": 1}}]})
    dobro = Tool("dobro", "dobra x", Args, lambda a: str(a.x * 2))
    with pytest.raises(AgentOutputError):
        ReActRunner(llm, caller="teste", max_steps=3).run("sys", "tarefa", [dobro], Saida)
    assert "ÚLTIMO PASSO" in llm.requests[-1].messages[-1].content


def test_conteudo_descarta_ids_inventados_e_entrega_texto_original(kb):
    passos = [
        {"pensamento": "", "acao": "busca_hibrida", "entrada": {"consulta": "vértice da parábola função quadrática"}},
        react_final({"trechos": ["id-inventado", "wb-quad"], "questoes_referencia": ["Q1"]}),
    ]
    agente = AgenteConteudo(FakeChatModel({"conteudo": passos}), kb, k=2, min_score=0.01)
    pacote = agente.buscar("ensinar função quadrática", Topico.FUNCOES)

    ids = [t.doc_id for t in pacote.trechos]
    assert ids[0] == "wb-quad" and "id-inventado" not in ids
    assert "wb-mru" not in ids  # filtro por tópico
    assert pacote.trechos[0].texto == kb.get("wb-quad").texto
    assert [q.doc_id for q in pacote.questoes_referencia] == ["Q1"]


def test_diagnosticador_cobre_conceitos_e_cai_para_fallback(banco):
    diag = Diagnosticador(
        FakeChatModel({"diagnosticador": ["inválido"]}), UsageLedgerCallback(), max_perguntas=4, max_repairs=0
    )
    sondagem = diag.selecionar_sondagem(Topico.FUNCOES, banco.find(Topico.FUNCOES))
    conceitos = [c for q in sondagem for c in q.conceitos]
    assert len(sondagem) == 4 and len(set(conceitos)) == 4  # cada pergunta cobre um conceito novo

    evidencias = [
        Evidencia(item_id="a", conceitos=["funcao_afim"], dificuldade=Dificuldade.MEDIA, correta=True),
        Evidencia(item_id="b", conceitos=["funcao_quadratica"], dificuldade=Dificuldade.MEDIA, correta=False),
    ]
    resultado = diag.estimar(PerfilAluno.novo("x"), Topico.FUNCOES, evidencias)
    assert resultado.conceitos_dominados == ["funcao_afim"]
    assert resultado.lacunas == ["funcao_quadratica"]
    assert resultado.nivel == NivelDominio.INICIANTE


def test_corretor_tem_corretude_deterministica():
    llm = FakeChatModel({"corretor": [{"nota": 0.9, "feedback": "ok", "erros": []}]})
    item = Exercicio(id="e1", topico=Topico.FUNCOES, enunciado="?", alternativas=list("abcde"), gabarito="C")
    # o LLM dá nota alta, mas a alternativa está errada: `correta` não vem do LLM
    correcao = Corretor(llm, UsageLedgerCallback()).corrigir(
        item, RespostaAluno(item_id="e1", escolha="A", resolucao="...")
    )
    assert correcao.correta is False and correcao.nota == 0.9

    # acerto sem resolução não gasta chamada de LLM
    Corretor(llm, UsageLedgerCallback()).corrigir(item, RespostaAluno(item_id="e1", escolha="c"))
    assert len(llm.requests) == 1
