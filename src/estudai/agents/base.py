"""Base `Agent` class and other related functions. Supports both agents who are simple LLM calls and ReAct agents.

Has retrying on simple agents and max retries on ReAct agents. Features like temperature are decoupled,
and should be the set on the passed model.
"""

import json
from typing import Any, TypeVar

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, StructuredOutputError, ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

from estudai.domain.mensagens import Papel

T = TypeVar("T", bound=BaseModel)


def to_json(data: Any) -> str:
    """Dumps a python object to JSON. Supports dictionaries and Pydantic schemas"""

    if isinstance(data, BaseModel):
        return data.model_dump_json(indent=2)
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def run_config(caller: str) -> RunnableConfig:
    """Setups the `caller` metadata for LangChain callbacks."""

    return {"metadata": {"caller": caller}, "run_name": caller}


# Vale para todos os agentes: alguns modelos escrevem português sem acentos
# ("voce ja domina") se isso não for pedido explicitamente.
REGRA_IDIOMA = (
    "\n\nEscreva todo texto em português do Brasil, com acentuação e ortografia corretas "
    '(ex.: "você", "já", "função"), inclusive dentro dos campos do JSON.'
)


def com_regra_idioma(system: str) -> str:
    return system + REGRA_IDIOMA


class AgentOutputError(RuntimeError):
    """O agente não produziu saída válida dentro do limite de reparos."""


class Agent:
    papel: Papel

    def __init__(self, model: BaseChatModel, *, max_repairs: int = 2) -> None:
        self._model = model
        self._max_repairs = max_repairs
        self._structured: dict[type[BaseModel], Any] = {}
        self._tool_agents: dict[tuple[Any, ...], Any] = {}

    def _config(self) -> RunnableConfig:
        return run_config(self.papel.value)

    def _structured_runnable(self, schema: type[T]) -> Any:
        """Get a runnable model that returns structured outputs"""

        runnable = self._structured.get(schema)
        if runnable is None:
            runnable = self._model.with_structured_output(schema, include_raw=True)
            self._structured[schema] = runnable
        return runnable

    def _get_raw_content(self, raw: BaseMessage) -> str:
        """Extract the raw message content, including when the response comes from tool calls (on structured output)"""
        if isinstance(raw.content, str) and raw.content.strip():
            return raw.content

        # Structured output on many providers can be returned as a function call, so extract that
        tool_calls = getattr(raw, "tool_calls", None) or []
        if tool_calls:
            return json.dumps([tc.get("args", {}) for tc in tool_calls], ensure_ascii=False)

        return str(raw.content)

    def _ask(self, system: str, prompt: str, schema: type[T]) -> T:
        """Calls the stored model with structured output, while retrying.

        On error, the appended message is appended as a `HumanMessage` for compatibility.
        """
        runnable = self._structured_runnable(schema)
        parser = PydanticOutputParser(pydantic_object=schema)
        messages: list[BaseMessage] = [SystemMessage(com_regra_idioma(system)), HumanMessage(prompt)]

        error = None
        for _ in range(self._max_repairs + 1):
            result = runnable.invoke(messages, config=self._config())
            if result["parsed"] is not None:
                return result["parsed"]

            error = result["parsing_error"]
            messages.append(
                HumanMessage(
                    "Sua saída anterior foi:\n"
                    f"{self._get_raw_content(result['raw'])}\n\n"
                    f"Sua saída é inválida: {error}\n"
                    f"Siga este formato:\n{parser.get_format_instructions()}\n"
                    "Corrija e responda novamente."
                )
            )
        raise AgentOutputError(f"{self.papel.value}: saída inválida após reparos: {error}")

    def _response_strategy(self, schema: type[BaseModel]) -> Any:
        """Como o agente de ferramentas obtém a resposta final estruturada.

        Por padrão, o LangChain usa `ToolStrategy` (uma ferramenta de resposta com `tool_choice="any"`)
        em modelos que ele não reconhece, e alguns provedores recusam isso: só aceitam
        `tool_choice="auto"`. Por isso a estratégia segue o `structured_output_method` do modelo:
        JSON Schema nativo → `ProviderStrategy`; `function_calling` → `ToolStrategy`.
        """
        metodo = getattr(self._model, "structured_output_method", None)
        if metodo is None:
            return schema  # modelo sem essa configuração (ex.: testes): escolha do LangChain
        return ToolStrategy(schema) if metodo == "function_calling" else ProviderStrategy(schema)

    def _run_tool_agent(
        self,
        prompt,
        tools,
        system_prompt,
        response_schema,
        max_steps=6,
        *,
        configurable: dict[str, Any] | None = None,
    ):
        """Calls a ReAct model, with a `max_steps` value"""

        # Check created agent cache
        key = (response_schema, system_prompt, tuple(t.name for t in tools))
        agent = self._tool_agents.get(key)
        if agent is None:
            agent = create_agent(
                self._model,
                tools=tools,
                system_prompt=com_regra_idioma(system_prompt),
                response_format=self._response_strategy(response_schema),
            )
            self._tool_agents[key] = agent

        config = {**self._config(), "recursion_limit": 2 * max_steps + 1}
        if configurable is not None:
            config["configurable"] = configurable

        try:
            result = agent.invoke({"messages": [HumanMessage(prompt)]}, config=config)
        except GraphRecursionError as exc:
            raise AgentOutputError(f"{self.papel.value}: limite de passos excedido") from exc
        except StructuredOutputError as exc:  # resposta final fora do schema: falha do agente, não do ciclo
            raise AgentOutputError(f"{self.papel.value}: resposta final inválida: {exc}") from exc
        response = result.get("structured_response")
        if response is None:
            raise AgentOutputError(f"{self.papel.value}: sem resposta estruturada")
        return response
