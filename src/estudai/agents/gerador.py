"""Agente Gerador de Exercícios (modelo de fronteira, LangChain + ferramentas).

Gera um item por vez para um conceito/dificuldade planejados. Ferramentas:
- `consultar_conteudo`: aciona o Agente de Conteúdo (questões de referência e material);
- `executar_python`: sandbox para conferir contas (sympy disponível).
O item ainda passa pelo painel de resolutores antes de ser entregue (application.validacao).
"""

from typing import Any, Literal
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError, model_validator

from estudai.agents.base import Agent, AgentOutputError, to_json
from estudai.agents.conteudo import AgenteConteudo
from estudai.agents.tools import call_conteudo_agent_tool, python_tool
from estudai.domain.curriculum import Topico
from estudai.domain.figura import PedidoFigura
from estudai.domain.latex import ESPECIFICACAO_LATEX, para_inline
from estudai.domain.mensagens import Papel
from estudai.domain.planejamento import ItemPlanejado
from estudai.domain.questoes import LETRAS, Exercicio, OrigemItem
from estudai.ports.tools import CodeExecutor

SYSTEM = (
    r"""Você é o Agente Gerador de Exercícios de um sistema de tutoria para vestibulares
brasileiros. Crie UMA questão objetiva inédita no estilo ENEM/Fuvest/Unicamp.
Requisitos:
- Exatamente 5 alternativas, apenas UMA correta; distratores plausíveis, baseados em erros típicos.
- Sem pistas gramaticais ou de tamanho entre alternativas. Enunciado claro e autocontido.
- Adeque ao conceito-alvo, à dificuldade e às lacunas do aluno.
- Confira as contas com `executar_python` antes de finalizar.
- Se a resposta for numérica, preencha `codigo_verificacao` com um script Python que faça
  `print()` APENAS do valor correto (número ou expressão sympy), para verificação automática.
- Use `consultar_conteudo` para se inspirar em questões reais e cite os ids em `referencias`.
- `alternativas` contém só o texto de cada opção, sem "A)" etc.
- `gabarito` é a LETRA da alternativa correta (A, B, C, D ou E), nunca o texto da resposta.
- Na `resolucao`, NÃO cite letras de alternativas (elas serão reordenadas): refira-se ao valor ou ao texto.
- TODA a matemática em `enunciado`, `alternativas` e `resolucao` deve seguir a notação abaixo."""
    + ESPECIFICACAO_LATEX
)

_REGRA_FIGURAS = """- Figuras: use no máximo {max_figuras} figura(s). Inclua pelo menos uma figura quando o
  conceito-alvo for visual (por exemplo: gráficos de funções, geometria, cinemática, forças,
  probabilidade ou diagramas de fluxo); não use figuras decorativas. Para cada uma, preencha
  `figuras` com `tipo` grafico (matplotlib) ou diagrama (Mermaid). A `descricao` precisa ser
  detalhada o suficiente para quem só lê o texto (corretores futuros verão somente o texto), não
  pode revelar a resposta e o enunciado com as legendas deve continuar resolvível sem a imagem."""


class RascunhoExercicio(BaseModel):
    enunciado: str
    alternativas: list[str] = Field(min_length=5, max_length=5)
    gabarito: Literal["A", "B", "C", "D", "E"] = Field(description="LETRA da alternativa correta, não o texto")
    resolucao: str
    codigo_verificacao: str | None = None
    referencias: list[str] = Field(default_factory=list)
    figuras: list[PedidoFigura] = Field(
        default_factory=list,
        description=(
            "Figuras que apoiam o enunciado. Inclua uma quando o conceito for visual (gráficos, "
            "geometria, cinemática, forças, probabilidade). A descrição deve bastar para resolver a "
            "questão sem a imagem e não pode revelar a resposta."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _gabarito_como_letra(cls, dados: Any) -> Any:
        """Rede de segurança: modelos às vezes devolvem o texto da alternativa correta no lugar da letra."""
        if not isinstance(dados, dict):
            return dados
        gabarito, alternativas = str(dados.get("gabarito", "")).strip(), dados.get("alternativas") or []
        if gabarito.upper() in LETRAS[: len(alternativas)]:
            return {**dados, "gabarito": gabarito.upper()}
        normalizar = lambda t: " ".join(str(t).lower().split())  # noqa: E731
        iguais = [i for i, alt in enumerate(alternativas) if normalizar(alt) == normalizar(gabarito)]
        return {**dados, "gabarito": LETRAS[iguais[0]]} if len(iguais) == 1 else dados


class GeradorExercicios(Agent):
    papel = Papel.GERADOR_EXERCICIOS

    def __init__(
        self,
        model: BaseChatModel,
        conteudo: AgenteConteudo,
        executor: CodeExecutor,
        *,
        usage_cb: BaseCallbackHandler | None = None,
        max_steps: int = 6,
        figuras: bool = True,
        max_figuras: int = 1,
        **kw,
    ) -> None:
        super().__init__(model, usage_cb, **kw)

        self._max_steps = max_steps
        self._max_figuras = max_figuras
        self._system = SYSTEM + ("\n\n" + _REGRA_FIGURAS.format(max_figuras=max_figuras) if figuras else "")
        self._tools = [python_tool(executor), call_conteudo_agent_tool(conteudo)]

    def gerar(self, perfil: dict, topico: Topico, plano: ItemPlanejado, evitar: list[str]) -> Exercicio:
        prompt = (
            f"Tópico: {topico.value}\nConceito-alvo: {plano.conceito}\n"
            f"Dificuldade-alvo: {plano.dificuldade.value}\n"
            f"Perfil do aluno:\n{to_json(perfil)}\n"
            "Não repita estes enunciados já usados:\n" + ("\n".join(f"- {e[:160]}" for e in evitar) or "(nenhum)")
        )

        rascunho = self._run_tool_agent(
            prompt,
            self._tools,
            self._system,
            RascunhoExercicio,
            self._max_steps,
            configurable={"topico": topico},
        )
        if len(rascunho.figuras) > self._max_figuras:
            rascunho = rascunho.model_copy(update={"figuras": rascunho.figuras[: self._max_figuras]})
        try:
            # embaralha antes da validação: o painel valida a mesma ordem que o aluno verá
            return self._montar(rascunho, topico, plano).com_alternativas_embaralhadas()
        except ValidationError as exc:  # rascunho incoerente: descarta o item, não o ciclo
            raise AgentOutputError(f"{self.papel.value}: rascunho inválido: {exc}") from exc

    @staticmethod
    def _montar(rascunho: RascunhoExercicio, topico: Topico, plano: ItemPlanejado) -> Exercicio:
        return Exercicio(
            id=f"gen-{uuid4().hex[:8]}",
            topico=topico,
            conceitos=[plano.conceito],
            dificuldade_alvo=plano.dificuldade,
            enunciado=para_inline(rascunho.enunciado),
            alternativas=[para_inline(a) for a in rascunho.alternativas],
            gabarito=rascunho.gabarito,
            resolucao=para_inline(rascunho.resolucao),
            origem=OrigemItem.GERADO,
            referencias=rascunho.referencias,
            codigo_verificacao=rascunho.codigo_verificacao,
            figuras=rascunho.figuras,
            descricoes_imagens=[figura.descricao for figura in rascunho.figuras],
        )
