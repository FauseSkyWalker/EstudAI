"""Infraestrutura comum dos agentes: saídas estruturadas validadas por schema.

Quando o modelo devolve uma saída inválida, o agente reenvia o pedido com a mensagem de
erro de validação como instrução de correção (Seção 4), até `max_repairs` vezes.
"""

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from estudai.domain.mensagens import Papel
from estudai.ports.llm import ChatMessage, LLMClient, LLMRequest

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class AgentOutputError(RuntimeError):
    """O agente não produziu saída válida dentro do limite de reparos."""


def extract_json(text: str) -> str:
    """Extrai o objeto JSON de uma resposta (tolera cercas markdown e texto ao redor)."""
    if match := _FENCE.search(text):
        text = match.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("nenhum objeto JSON encontrado na resposta")
    return text[start : end + 1]


def schema_instructions(schema: type[BaseModel]) -> str:
    return (
        "Responda APENAS com um objeto JSON válido (sem markdown, sem texto extra) "
        "que siga exatamente este JSON Schema:\n"
        f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
    )


def to_json(data: Any) -> str:
    if isinstance(data, BaseModel):
        return data.model_dump_json(indent=2)
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


class Agent:
    papel: Papel

    def __init__(self, llm: LLMClient, *, max_repairs: int = 2, temperature: float = 0.2) -> None:
        self._llm = llm
        self._max_repairs = max_repairs
        self._temperature = temperature

    def _ask(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        *,
        temperature: float | None = None,
    ) -> T:
        messages = [ChatMessage(role="user", content=f"{prompt}\n\n{schema_instructions(schema)}")]
        erro: Exception | None = None
        for _ in range(self._max_repairs + 1):
            response = self._llm.complete(
                LLMRequest(
                    system=system,
                    messages=messages,
                    json_schema=schema.model_json_schema(),
                    temperature=self._temperature if temperature is None else temperature,
                    caller=self.papel.value,
                )
            )
            try:
                return schema.model_validate_json(extract_json(response.text))
            except (ValidationError, ValueError) as exc:
                erro = exc
                messages += [
                    ChatMessage(role="assistant", content=response.text),
                    ChatMessage(
                        role="user",
                        content=f"Sua saída é inválida:\n{exc}\nCorrija e responda apenas o JSON.",
                    ),
                ]
        raise AgentOutputError(f"{self.papel.value}: saída inválida após reparos: {erro}")
