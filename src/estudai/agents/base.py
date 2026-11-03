"""Infraestrutura comum dos agentes: saídas estruturadas validadas por schema.

Quando o modelo devolve uma saída inválida, o agente reenvia o pedido com a mensagem de
erro de validação como instrução de correção (Seção 4), até `max_repairs` vezes.

Temperatura, retries e callbacks de uso são propriedades do modelo, configuradas na raiz de
composição (`bootstrap.py`); o agente só identifica quem chamou (`metadata.caller`).
"""

import json
from typing import Any, TypeVar

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

from estudai.domain.mensagens import Papel

T = TypeVar("T", bound=BaseModel)


def to_json(data: Any) -> str:
    if isinstance(data, BaseModel):
        return data.model_dump_json(indent=2)
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def run_config(caller: str, usage_cb: BaseCallbackHandler | None = None) -> RunnableConfig:
    """`caller` vai nos metadados da execução: é por ele que a telemetria separa o custo por agente."""
    config: RunnableConfig = {"metadata": {"caller": caller}, "run_name": caller}
    if usage_cb is not None:
        config["callbacks"] = [usage_cb]
    return config


class AgentOutputError(RuntimeError):
    """O agente não produziu saída válida dentro do limite de reparos."""


class Agent:
    papel: Papel

    def __init__(
        self,
        model: BaseChatModel,
        usage_cb: BaseCallbackHandler | None = None,
        *,
        max_repairs: int = 2,
    ) -> None:
        self._model = model
        self._usage_cb = usage_cb
        self._max_repairs = max_repairs

    def _config(self) -> RunnableConfig:
        return run_config(self.papel.value, self._usage_cb)

    def _ask(self, system: str, prompt: str, schema: type[T]) -> T:
        messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(prompt)]
        runnable = self._model.with_structured_output(schema, include_raw=True)

        erro = None
        for _ in range(self._max_repairs + 1):
            result = runnable.invoke(messages, config=self._config())
            if result["parsed"] is not None:
                return result["parsed"]
            erro = result["parsing_error"]
            messages += [
                _sem_tool_calls(result["raw"]),
                HumanMessage(f"Sua saída é inválida: {erro}\nCorrija e responda novamente."),
            ]
        raise AgentOutputError(f"{self.papel.value}: saída inválida após reparos: {erro}")

    def _run_tool_agent(self, prompt, tools, system_prompt, response_schema, max_steps=6):
        agent = create_agent(self._model, tools=tools, system_prompt=system_prompt, response_format=response_schema)
        config = {**self._config(), "recursion_limit": 2 * max_steps + 1}

        try:
            result = agent.invoke({"messages": [HumanMessage(prompt)]}, config=config)
        except GraphRecursionError as exc:
            raise AgentOutputError(f"{self.papel.value}: limite de passos excedido") from exc
        response = result.get("structured_response")
        if response is None:
            raise AgentOutputError(f"{self.papel.value}: sem resposta estruturada")
        return response


def _sem_tool_calls(raw: BaseMessage) -> BaseMessage:
    """Com `method="function_calling"`, a resposta crua traz `tool_calls`; reenviá-la sem os
    `ToolMessage` correspondentes é rejeitado pela API. Converte para texto."""
    tool_calls = getattr(raw, "tool_calls", None)
    if not tool_calls:
        return raw
    return AIMessage(content=json.dumps([tc["args"] for tc in tool_calls], ensure_ascii=False))
