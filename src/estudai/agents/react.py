"""Laço ReAct (S. Yao et al., 2023) agnóstico de fornecedor.

Cada passo é um JSON `{pensamento, acao, entrada}`. Em vez de depender de function calling
nativo (que difere entre fornecedores), as ferramentas são descritas no prompt e as
observações voltam como mensagens. Erros de ferramenta e argumentos inválidos viram
observações, permitindo que o modelo se corrija (fallback estruturado da Seção 4).
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, ValidationError

from estudai.agents.base import AgentOutputError
from estudai.ports.llm import ChatMessage, LLMClient, LLMRequest

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> str:
    """Extrai o objeto JSON de uma resposta (tolera cercas markdown e texto ao redor)."""
    if match := _FENCE.search(text):
        text = match.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("nenhum objeto JSON encontrado na resposta")
    return text[start : end + 1]

T = TypeVar("T", bound=BaseModel)
A = TypeVar("A", bound=BaseModel)

FINAL = "final"


@dataclass(frozen=True)
class Tool(Generic[A]):
    name: str
    description: str
    args_model: type[A]
    fn: Callable[[A], str]

    def spec(self) -> str:
        args = json.dumps(self.args_model.model_json_schema().get("properties", {}), ensure_ascii=False)
        return f"- {self.name}: {self.description}\n  entrada: {args}"


class ReActStep(BaseModel):
    pensamento: str
    acao: str
    entrada: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ReActTrace:
    steps: list[tuple[ReActStep, str]] = field(default_factory=list)

    @property
    def tool_calls(self) -> int:
        return sum(1 for step, _ in self.steps if step.acao != FINAL)


_INSTRUCOES = """
Você opera no padrão ReAct. A cada passo, responda APENAS com um JSON:
{{"pensamento": "<raciocínio breve>", "acao": "<nome da ferramenta ou 'final'>", "entrada": {{...}}}}

Ferramentas disponíveis:
{tools}

Quando terminar, use "acao": "final" e coloque em "entrada" a resposta final, que deve seguir
este JSON Schema:
{output_schema}
"""


class ReActRunner:
    def __init__(self, llm: LLMClient, *, caller: str, max_steps: int = 6, max_obs_chars: int = 3000):
        self._llm = llm
        self._caller = caller
        self._max_steps = max_steps
        self._max_obs_chars = max_obs_chars

    def run(
        self,
        system: str,
        task: str,
        tools: list[Tool],
        output_schema: type[T],
        *,
        temperature: float = 0.3,
    ) -> tuple[T, ReActTrace]:
        by_name = {t.name: t for t in tools}
        system_full = system + _INSTRUCOES.format(
            tools="\n".join(t.spec() for t in tools),
            output_schema=json.dumps(output_schema.model_json_schema(), ensure_ascii=False),
        )
        messages = [ChatMessage(role="user", content=task)]
        trace = ReActTrace()

        for passo in range(self._max_steps):
            response = self._llm.complete(
                LLMRequest(
                    system=system_full,
                    messages=messages,
                    json_schema=ReActStep.model_json_schema(),
                    temperature=temperature,
                    caller=self._caller,
                )
            )
            messages.append(ChatMessage(role="assistant", content=response.text))
            try:
                step = ReActStep.model_validate_json(_extract_json(response.text))
            except (ValidationError, ValueError) as exc:
                observacao = f"Formato inválido ({exc}). Responda com o JSON {{pensamento, acao, entrada}}."
            else:
                if step.acao == FINAL:
                    try:
                        resultado = output_schema.model_validate(step.entrada)
                    except ValidationError as exc:
                        observacao = f"Resposta final inválida: {exc}"
                    else:
                        trace.steps.append((step, ""))
                        return resultado, trace
                else:
                    observacao = self._executar(by_name, step)
                trace.steps.append((step, observacao))

            restantes = self._max_steps - passo - 1
            aviso = " ÚLTIMO PASSO: responda agora com acao='final'." if restantes == 1 else ""
            messages.append(
                ChatMessage(
                    role="user",
                    content=f"Observação: {observacao[: self._max_obs_chars]}\n(Passos restantes: {restantes}){aviso}",
                )
            )
        raise AgentOutputError(f"{self._caller}: ReAct sem resposta final em {self._max_steps} passos")

    @staticmethod
    def _executar(by_name: dict[str, Tool], step: ReActStep) -> str:
        tool = by_name.get(step.acao)
        if tool is None:
            return f"Ferramenta '{step.acao}' não existe. Use uma de {sorted(by_name)} ou 'final'."
        try:
            return tool.fn(tool.args_model.model_validate(step.entrada))
        except ValidationError as exc:
            return f"Argumentos inválidos para {tool.name}: {exc}"
        except Exception as exc:  # noqa: BLE001 - falha de ferramenta vira observação
            return f"Erro ao executar {tool.name}: {type(exc).__name__}: {exc}"
