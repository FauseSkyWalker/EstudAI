"""Infraestrutura comum dos agentes: saídas estruturadas validadas por schema.

Quando o modelo devolve uma saída inválida, o agente reenvia o pedido com a mensagem de
erro de validação como instrução de correção (Seção 4), até `max_repairs` vezes.
"""

import json
from typing import Any, TypeVar

from langchain.chat_models import BaseChatModel
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel

from estudai.domain.mensagens import Papel

T = TypeVar("T", bound=BaseModel)


def to_json(data: Any) -> str:
    if isinstance(data, BaseModel):
        return data.model_dump_json(indent=2)
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


class AgentOutputError(RuntimeError):
    """O agente não produziu saída válida dentro do limite de reparos."""


class Agent:
    papel: Papel

    def __init__(self, model: BaseChatModel, usage_cb: BaseCallbackHandler, *, max_repairs: int = 2):
        self._model = model
        self._usage_cb = usage_cb
        self._max_repairs = max_repairs

    def _config(self):
        return {"callbacks": [self._usage_cb], "metadata": {"caller": self.papel.value}}

    def _ask(self, system: str, prompt: str, schema: type[T]) -> T:
        messages = [SystemMessage(system), HumanMessage(prompt)]
        runnable = self._model.with_structured_output(schema, include_raw=True)

        for _ in range(self._max_repairs + 1):
            result = runnable.invoke(messages, config=self._config())

            if result["parsed"] is not None:
                return result["parsed"]
            messages += [
                result["raw"],
                HumanMessage(f"Sua saída é inválida: {result['parsing_error']}\nCorrija e responda novamente."),
            ]

        raise AgentOutputError(f"{self.papel.value}: saída inválida após reparos")
