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
from estudai.domain.mensagens import Papel
from estudai.domain.planejamento import ItemPlanejado
from estudai.domain.questoes import LETRAS, Exercicio, OrigemItem
from estudai.ports.tools import CodeExecutor

SYSTEM = """Você é o Agente Gerador de Exercícios de um sistema de tutoria para vestibulares
brasileiros. Crie UMA questão objetiva inédita no estilo ENEM/Fuvest/Unicamp.
Requisitos:
- Exatamente 5 alternativas, apenas UMA correta; distratores plausíveis, baseados em erros típicos.
- Sem pistas gramaticais ou de tamanho entre alternativas; enunciado claro e autocontido (sem figuras).
- Adeque ao conceito-alvo, à dificuldade e às lacunas do aluno.
- Confira as contas com `executar_python` antes de finalizar.
- Se a resposta for numérica, preencha `codigo_verificacao` com um script Python que faça
  `print()` APENAS do valor correto (número ou expressão sympy), para verificação automática.
- Use `consultar_conteudo` para se inspirar em questões reais e cite os ids em `referencias`.
- `alternativas` contém só o texto de cada opção, sem "A)" etc.
- `gabarito` é a LETRA da alternativa correta (A, B, C, D ou E), nunca o texto da resposta.
"""


class RascunhoExercicio(BaseModel):
    enunciado: str
    alternativas: list[str] = Field(min_length=5, max_length=5)
    gabarito: Literal["A", "B", "C", "D", "E"] = Field(description="LETRA da alternativa correta, não o texto")
    resolucao: str
    codigo_verificacao: str | None = None
    referencias: list[str] = Field(default_factory=list)

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
        **kw,
    ) -> None:
        super().__init__(model, usage_cb, **kw)

        self._max_steps = max_steps
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
            SYSTEM,
            RascunhoExercicio,
            self._max_steps,
            configurable={"topico": topico},
        )
        try:
            return self._montar(rascunho, topico, plano)
        except ValidationError as exc:  # rascunho incoerente: descarta o item, não o ciclo
            raise AgentOutputError(f"{self.papel.value}: rascunho inválido: {exc}") from exc

    @staticmethod
    def _montar(rascunho: RascunhoExercicio, topico: Topico, plano: ItemPlanejado) -> Exercicio:
        return Exercicio(
            id=f"gen-{uuid4().hex[:8]}",
            topico=topico,
            conceitos=[plano.conceito],
            dificuldade_alvo=plano.dificuldade,
            enunciado=rascunho.enunciado,
            alternativas=rascunho.alternativas,
            gabarito=rascunho.gabarito,
            resolucao=rascunho.resolucao,
            origem=OrigemItem.GERADO,
            referencias=rascunho.referencias,
            codigo_verificacao=rascunho.codigo_verificacao,
        )
