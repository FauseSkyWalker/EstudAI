import pytest
from fakes import FakeChatModel, structured_final, tool_call
from langchain_core.tools import tool
from pydantic import BaseModel

from estudai.agents.base import Agent, AgentOutputError
from estudai.agents.conteudo import AgenteConteudo
from estudai.agents.corretor import Corretor
from estudai.agents.diagnosticador import Diagnosticador, Evidencia
from estudai.agents.gerador import GeradorExercicios
from estudai.agents.resolutor import Resolutor
from estudai.agents.resumo import AgenteResumo
from estudai.agents.tutor import Tutor
from estudai.domain.correcao import RespostaAluno
from estudai.domain.curriculum import Disciplina, Topico
from estudai.domain.material import PacoteMaterial, TipoDocumento, TrechoRecuperado
from estudai.domain.mensagens import Papel
from estudai.domain.perfil import NivelDominio, PerfilAluno
from estudai.domain.planejamento import ItemPlanejado, ObjetivoSessao
from estudai.domain.questoes import Dificuldade, Exercicio, OrigemItem
from estudai.domain.tutoria import TranscricaoTutoria
from estudai.infrastructure.llm.callbacks import UsageLedgerCallback
from estudai.ports.knowledge import SearchMode
from estudai.ports.tools import ExecutionResult


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


class FerramentaTeste(Agent):
    papel = Papel.TUTOR

    def pedir(self, tools, *, max_steps: int = 4) -> Saida:
        return self._run_tool_agent("tarefa", tools, "sys", Saida, max_steps)


@tool
def dobro(x: int) -> str:
    """Dobra x."""
    return str(x * 2)


def test_tool_agent_usa_ferramenta_e_retorna_estruturado():
    llm = FakeChatModel(
        {
            "tutor": [
                tool_call("dobro", {"x": 21}, "call-dobro"),
                structured_final("Saida", {"valor": 42}, "call-final"),
            ]
        }
    )
    assert FerramentaTeste(llm).pedir([dobro]).valor == 42


def test_tool_agent_respeita_limite_de_passos():
    llm = FakeChatModel({"tutor": [tool_call("dobro", {"x": 1}, "call-dobro")]})
    with pytest.raises(AgentOutputError):
        FerramentaTeste(llm).pedir([dobro], max_steps=1)


class _ConteudoStub:
    def __init__(self) -> None:
        self.calls = []

    def buscar(self, pedido, topico=None, incluir_questoes=True):
        self.calls.append((pedido, topico, incluir_questoes))
        return PacoteMaterial(
            pedido=pedido,
            trechos=[
                TrechoRecuperado(
                    doc_id="wb-quad",
                    tipo=TipoDocumento.MATERIAL,
                    fonte="wikibooks",
                    texto="A função quadrática tem vértice em x = -b/2a.",
                    score=0.9,
                )
            ],
        )


class _ExecutorStub:
    def __init__(self) -> None:
        self.calls = []

    def run(self, code, timeout_s=None):
        self.calls.append((code, timeout_s))
        return ExecutionResult(ok=True, stdout="5")


def test_gerador_usa_ferramentas_e_entrega_rascunho():
    passos = [
        tool_call("consultar_conteudo", {"pedido": "função quadrática"}, "call-conteudo"),
        tool_call("executar_python", {"code": "print(2**2 + 1)", "timeout_s": 5}, "call-python"),
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
        ),
    ]
    conteudo, executor = _ConteudoStub(), _ExecutorStub()
    gerador = GeradorExercicios(FakeChatModel({"gerador_exercicios": passos}), conteudo, executor)
    plano = ItemPlanejado(conceito="funcao_quadratica", dificuldade=Dificuldade.MEDIA)

    item = gerador.gerar({}, Topico.FUNCOES, plano, [])

    assert conteudo.calls == [("função quadrática", Topico.FUNCOES, True)]
    assert executor.calls == [("print(2**2 + 1)", 5.0)]
    assert item.topico == Topico.FUNCOES
    assert item.origem == OrigemItem.GERADO
    assert item.alternativa(item.gabarito) == "5"  # alternativas embaralhadas: vale o conteúdo
    assert item.codigo_verificacao == "print(2**2 + 1)"


def test_gerador_normaliza_delimitadores_de_latex():
    from estudai.agents.gerador import GeradorExercicios, RascunhoExercicio

    rascunho = RascunhoExercicio(
        enunciado=r"Seja $x^2$ e R$ 10, calcule \[y\].",
        alternativas=["$1$", "$2$", "$3$", "$4$", "$5$"],
        gabarito="A",
        resolucao=r"\(x^2\)",
    )
    plano = ItemPlanejado(conceito="funcao_quadratica", dificuldade=Dificuldade.MEDIA)

    item = GeradorExercicios._montar(rascunho, Topico.FUNCOES, plano)

    assert r"\(x^2\)" in item.enunciado
    assert "R$ 10" in item.enunciado  # cifra não vira matemática
    assert r"\(y\)" in item.enunciado
    assert item.alternativas[0] == r"\(1\)"


def test_conteudo_usa_ferramenta_e_descarta_ids_inventados(kb, monkeypatch):
    passos = [
        tool_call("busca_hibrida", {"consulta": "vértice da parábola função quadrática"}, "call-search"),
        structured_final(
            "SelecaoConteudo",
            {"trechos": ["id-inventado", "wb-quad"], "questoes_referencia": ["Q1"]},
            "call-final",
        ),
    ]
    chamadas = []
    busca_original = kb.search

    def search(*args, **kwargs):
        chamadas.append((args, kwargs))
        return busca_original(*args, **kwargs)

    monkeypatch.setattr(kb, "search", search)
    agente = AgenteConteudo(FakeChatModel({"conteudo": passos}), kb, k=2)
    pacote = agente.buscar("ensinar função quadrática", Topico.FUNCOES)

    # a ferramenta de busca foi executada, não o fallback direto
    assert chamadas
    assert chamadas[0][0][0] == "vértice da parábola função quadrática"
    assert chamadas[0][0][1] == SearchMode.HIBRIDA
    assert chamadas[0][1]["topico"] == Topico.FUNCOES

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


def test_estrategia_de_resposta_segue_o_metodo_do_modelo():
    """A API da Meta (Muse) recusa tool_choice forçado; com JSON Schema nativo o agente usa ProviderStrategy."""
    from types import SimpleNamespace

    from langchain.agents.structured_output import ProviderStrategy, ToolStrategy

    def estrategia(modelo):
        return AgenteTeste(modelo)._response_strategy(Saida)

    assert isinstance(estrategia(SimpleNamespace(structured_output_method="json_schema")), ProviderStrategy)
    assert isinstance(estrategia(SimpleNamespace(structured_output_method="function_calling")), ToolStrategy)
    assert estrategia(SimpleNamespace()) is Saida  # sem configuração: escolha automática do LangChain


def test_rascunho_aceita_letra_e_converte_texto_da_alternativa():
    from pydantic import ValidationError

    from estudai.agents.gerador import RascunhoExercicio

    base = {"enunciado": "?", "alternativas": ["1", "(x - 1)/6", "3", "4", "5"], "resolucao": "."}
    assert RascunhoExercicio(**base, gabarito="c").gabarito == "C"
    assert RascunhoExercicio(**base, gabarito="(X - 1)/6").gabarito == "B"  # caso real visto com o Muse
    with pytest.raises(ValidationError):
        RascunhoExercicio(**base, gabarito="nenhuma")


def test_rascunho_incoerente_vira_falha_do_agente_e_nao_derruba_o_ciclo(monkeypatch):
    from estudai.agents.gerador import GeradorExercicios, RascunhoExercicio
    from estudai.domain.perfil import PerfilAluno
    from estudai.domain.planejamento import planejar_pratica

    gerador = GeradorExercicios(FakeChatModel({}), conteudo=None, executor=None)
    ruim = RascunhoExercicio.model_construct(
        enunciado="?", alternativas=["1", "2", "3", "4", "5"], gabarito="Z", resolucao=".", referencias=[]
    )
    monkeypatch.setattr(gerador, "_run_tool_agent", lambda *a, **k: ruim)
    plano = planejar_pratica(PerfilAluno.novo("x"), Topico.FUNCOES, 1)[0]
    with pytest.raises(AgentOutputError, match="rascunho inválido"):
        gerador.gerar({}, Topico.FUNCOES, plano, [])


def test_tutor_normaliza_latex_nas_saidas():
    llm = FakeChatModel(
        {
            ("tutor", "TurnoTutor"): [
                {
                    "avaliacao_resposta_anterior": r"Você usou $x^2$.",
                    "conceito_alvo": "funcao_quadratica",
                    "explicacao": r"O vértice é $x_v = -\frac{b}{2a}$.",
                    "pergunta_verificacao": r"Quanto vale $f(1)$?",
                }
            ]
        }
    )
    objetivo = ObjetivoSessao(
        topico=Topico.FUNCOES, conceitos_alvo=["funcao_quadratica"], nivel_aluno=None, descricao="ensinar"
    )
    turno = Tutor(llm, Disciplina.MATEMATICA).proximo_turno(
        {}, objetivo, PacoteMaterial(pedido="função quadrática"), TranscricaoTutoria(objetivo=objetivo)
    )

    assert turno.avaliacao_resposta_anterior == r"Você usou \(x^2\)."
    assert turno.explicacao == r"O vértice é \(x_v = -\frac{b}{2a}\)."
    assert turno.pergunta_verificacao == r"Quanto vale \(f(1)\)?"


def _system_do_tutor(**kwargs) -> str:
    llm = FakeChatModel(
        {
            ("tutor", "TurnoTutor"): [
                {"conceito_alvo": "funcao_quadratica", "explicacao": "x", "pergunta_verificacao": "y?"}
            ]
        }
    )
    objetivo = ObjetivoSessao(
        topico=Topico.FUNCOES, conceitos_alvo=["funcao_quadratica"], nivel_aluno=None, descricao="ensinar"
    )
    Tutor(llm, Disciplina.MATEMATICA, **kwargs).proximo_turno(
        {}, objetivo, PacoteMaterial(pedido="função quadrática"), TranscricaoTutoria(objetivo=objetivo)
    )
    return next(m.content for m in llm.requests[0].messages if m.type == "system")


def test_tutor_inclui_regra_de_figuras_por_padrao():
    assert "Figuras:" in _system_do_tutor()


def test_tutor_omite_regra_de_figuras_quando_desligadas():
    assert "Figuras:" not in _system_do_tutor(figuras=False)


def test_corretor_normaliza_latex_no_feedback_e_nos_erros():
    llm = FakeChatModel(
        {
            ("corretor", "AnaliseCorrecao"): [
                {
                    "nota": 0.3,
                    "feedback": r"Reveja $x^2$.",
                    "erros": [
                        {"tipo": "numerico", "conceito": "funcao_quadratica", "descricao": r"trocou $a$ por $b$"}
                    ],
                }
            ]
        }
    )
    item = Exercicio(id="e1", topico=Topico.FUNCOES, enunciado="?", alternativas=list("abcde"), gabarito="C")

    correcao = Corretor(llm).corrigir(item, RespostaAluno(item_id="e1", escolha="A"))

    assert correcao.feedback == r"Reveja \(x^2\)."
    assert correcao.erros[0].descricao == r"trocou \(a\) por \(b\)"


def test_resolutor_normaliza_latex_no_raciocinio():
    llm = FakeChatModel(
        {("painel_resolutores", "VotoResolutor"): [{"raciocinio": r"Como $x^2 = 4$, $x = 2$.", "resposta": "b"}]}
    )
    item = Exercicio(id="e1", topico=Topico.FUNCOES, enunciado="?", alternativas=list("abcde"), gabarito="B")

    voto = Resolutor(llm).resolver(item)

    assert voto.raciocinio == r"Como \(x^2 = 4\), \(x = 2\)."
    assert voto.resposta == "B"


def test_resumo_normaliza_latex_nos_campos():
    llm = FakeChatModel(
        {
            ("resumo", "ResumoEstudo"): [
                {
                    "titulo": "Função $f(x)$",
                    "pontos_chave": [r"vértice $x_v = -b/2a$"],
                    "formulas": [r"$x = \frac{-b}{2a}$"],
                    "erros_comuns": [r"esquecer $a \ne 0$"],
                    "proximos_passos": [r"praticar $f(1)$"],
                }
            ],
            ("resumo", "RelatorioPratica"): [
                {
                    "desempenho": r"Você acertou $70\%$.",
                    "pontos_fortes": [r"domina $x^2$"],
                    "pontos_a_melhorar": [r"rever $a$"],
                    "recomendacoes": [r"treine $f(2)$"],
                }
            ],
        }
    )
    objetivo = ObjetivoSessao(
        topico=Topico.FUNCOES, conceitos_alvo=["funcao_quadratica"], nivel_aluno=None, descricao="ensinar"
    )
    agente = AgenteResumo(llm)

    resumo = agente.resumir_tutoria(TranscricaoTutoria(objetivo=objetivo), PacoteMaterial(pedido="função quadrática"))
    relatorio = agente.resumir_pratica([], {}, {})

    assert resumo.titulo == r"Função \(f(x)\)"
    assert resumo.pontos_chave == [r"vértice \(x_v = -b/2a\)"]
    assert resumo.formulas == [r"\(x = \frac{-b}{2a}\)"]
    assert resumo.erros_comuns == [r"esquecer \(a \ne 0\)"]
    assert resumo.proximos_passos == [r"praticar \(f(1)\)"]
    assert relatorio.desempenho == r"Você acertou \(70\%\)."
    assert relatorio.pontos_fortes == [r"domina \(x^2\)"]
    assert relatorio.pontos_a_melhorar == [r"rever \(a\)"]
    assert relatorio.recomendacoes == [r"treine \(f(2)\)"]
