"""Agente Tutor por Disciplina (modelo de fronteira): uma instância para Matemática e outra para Física."""

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel

from estudai.agents.base import Agent, AgentOutputError, to_json
from estudai.agents.conteudo import AgenteConteudo
from estudai.agents.tools import call_conteudo_agent_tool
from estudai.domain.curriculum import Disciplina
from estudai.domain.latex import ESPECIFICACAO_LATEX, para_inline
from estudai.domain.material import PacoteMaterial
from estudai.domain.mensagens import Papel
from estudai.domain.planejamento import ObjetivoSessao
from estudai.domain.tutoria import TranscricaoTutoria, TurnoTutor

_ORIENTACOES = {
    Disciplina.MATEMATICA: (
        "Explique a ideia antes do procedimento, mostre um exemplo numérico curto e destaque "
        "armadilhas comuns de vestibular (sinais, domínio, casos particulares). "
        "Exemplos de conceitos visuais: gráficos de funções, parábolas com raízes e vértice, "
        "árvores de probabilidade, diagramas de Venn e a reta numérica."
    ),
    Disciplina.FISICA: (
        "Parta do fenômeno físico, explicite grandezas e unidades, indique o referencial e só "
        "então a equação. Verifique a coerência dimensional e a ordem de grandeza. "
        "Exemplos de conceitos visuais: gráficos sxt e vxt, trajetórias de lançamento, "
        "diagramas de corpo livre, planos inclinados e circuitos."
    ),
}

_REGRA_CONTEUDO = """- Consulta ao material: o material inicial está no pedido desta rodada. Se ele não cobrir o
  conceito-alvo ou o usuário explicitamente pedir material externo, use `consultar_conteudo` UMA vez; 
  depois responda com o que tiver. Cite os ids retornados entre colchetes, ex.: [wb-123]."""

_REGRA_FIGURAS = """- Figuras: o aluno vê imagens. Quando um conceito for visual, preencha `figura` com um pedido curto:
  - `tipo="grafico"` para gráficos quantitativos e desenhos físicos. Escreva código matplotlib
    (`plt` e `np` já estão importados; não use plt.show(), plt.savefig() nem arquivos), com eixos
    rotulados e unidades.
  - `tipo="diagrama"` para diagramas conceituais. Escreva fonte Mermaid (flowchart, mindmap, timeline,
    stateDiagram ou classDiagram), com rótulos curtos e sem links/cliques. Em diagramas Mermaid,
    NÃO use `$...$`, `$$...$$` nem LaTeX: escreva a matemática em texto simples e curto
    (ex.: `f(x) = ax + b`, `a > 0`, `x = -b/2a`). Coloque rótulos com parênteses, colchetes, chaves,
    vírgulas ou sinais entre aspas duplas, ex.: `F["resolver f(x) = valor"]`. Sem aspas o Mermaid
    interpreta `(` como forma. Se a fórmula for complexa, explique-a no texto da resposta.
  Use no máximo {max_figuras} figura(s) por rodada, com números coerentes com a explicação. As
  figuras e a descrição não podem revelar a resposta da pergunta de verificação. Deixe `figuras` vazio quando 
  o conceito não for visual ou quando a figura repetir uma anterior.
"""

SYSTEM = """Você é o Agente Tutor de {disciplina} de um sistema de tutoria para vestibulares brasileiros.
Conduza a sessão em rodadas curtas, em português claro, adaptadas ao nível do aluno.
{orientacao}
Regras:
- Ensine UM conceito-alvo por rodada e termine com UMA pergunta de verificação.
- Fundamente-se no material fornecido e cite os ids entre colchetes, ex.: [wb-123]. Não invente fatos.
- Avalie a resposta anterior do aluno antes de seguir; se errada, reexplique de outro jeito.
- Inclua em `conceitos_validados` apenas conceitos que o aluno demonstrou entender respondendo
  corretamente a uma pergunta de verificação nesta sessão.
- Marque `objetivo_atingido` somente quando todos os conceitos-alvo estiverem validados.
- Use `sessoes_anteriores` do perfil (memória das sessões passadas) para dar continuidade: na
  primeira rodada, retome em uma frase o que já foi visto; priorize os `proximos_passos` e os
  `erros_comuns`/`conceitos_com_erro` que se repetem; não reexplique do zero o que o aluno já
  dominou. Sem sessões anteriores, comece normalmente.
{figuras}
{conteudo}"""


class Tutor(Agent):
    papel = Papel.TUTOR

    def __init__(
        self,
        model: BaseChatModel,
        disciplina: Disciplina,
        conteudo: AgenteConteudo,
        *,
        usage_cb: BaseCallbackHandler | None = None,
        figuras: bool = True,
        max_figuras: int = 2,
        max_steps: int = 3,
        **kw,
    ):
        super().__init__(model, usage_cb, **kw)
        self.disciplina = disciplina
        self._max_steps = max_steps
        self._tools = [call_conteudo_agent_tool(conteudo)]
        regra_figuras = _REGRA_FIGURAS.format(max_figuras=max_figuras) if figuras else ""
        self._system = (
            SYSTEM.format(
                disciplina=disciplina.value,
                orientacao=_ORIENTACOES[disciplina],
                figuras=regra_figuras,
                conteudo=_REGRA_CONTEUDO,
            )
            + ESPECIFICACAO_LATEX
        )

    def proximo_turno(
        self,
        perfil: dict,
        objetivo: ObjetivoSessao,
        material: PacoteMaterial,
        transcricao: TranscricaoTutoria,
    ) -> TurnoTutor:
        historico = transcricao.como_texto() or "(início da sessão)"
        prompt = (
            f"Objetivo da sessão:\n{to_json(objetivo)}\n\n"
            f"Perfil do aluno:\n{to_json(perfil)}\n\n"
            f"Material de apoio:\n{material.como_contexto()}\n\n"
            f"Conversa até aqui:\n{historico}\n\n"
            f"Conceitos já validados: {transcricao.conceitos_validados}\n"
            "Produza a próxima rodada."
        )
        try:
            turno = self._run_tool_agent(
                prompt,
                self._tools,
                self._system,
                TurnoTutor,
                self._max_steps,
                configurable={"topico": objetivo.topico},
            )
        except AgentOutputError:
            # fallback - responde sem consultar a base caos
            turno = self._ask(
                self._system,
                prompt + "\n\nResponda usando apenas o material já fornecido; não consulte a base agora.",
                TurnoTutor,
            )
        return turno.model_copy(
            update={
                "avaliacao_resposta_anterior": (
                    para_inline(turno.avaliacao_resposta_anterior) if turno.avaliacao_resposta_anterior else None
                ),
                "explicacao": para_inline(turno.explicacao),
                "pergunta_verificacao": para_inline(turno.pergunta_verificacao),
            }
        )
