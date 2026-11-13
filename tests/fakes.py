"""Dublês de teste: LLM roteirizado e aluno roteirizado."""

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.outputs.chat_result import ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import PrivateAttr, ValidationError

from estudai.domain.correcao import RespostaAluno
from estudai.domain.figura import Figura, FiguraError, FormatoFigura, PedidoFigura
from estudai.domain.notificacao import Notificacao
from estudai.domain.questoes import ItemObjetivo
from estudai.domain.tutoria import TurnoTutor

Handler = Callable[..., str | dict[str, Any] | AIMessage]


@dataclass
class FakeRequest:
    """Registro do que o modelo recebeu (para inspeção nos testes)."""

    messages: list[Any]
    caller: str = ""
    schema_name: str | None = None


class FakeChatModel(BaseChatModel):
    """Modelo roteirizado por `caller` e, opcionalmente, por `(caller, schema)`.

    O handler pode ser:
    - uma string (JSON cru) ou um dict (serializado para JSON);
    - um `AIMessage` (útil para tool calls);
    - uma lista desses valores, consumida em ordem (o último se repete);
    - uma função que recebe `FakeRequest`.
    """

    handlers: dict[Any, Any]
    model_name: str = "fake"

    _counters: dict[Any, int] = PrivateAttr(default_factory=lambda: defaultdict(int))
    _requests: list[FakeRequest] = PrivateAttr(default_factory=list)

    def __init__(self, handlers: dict[Any, Any], **kwargs: Any) -> None:
        super().__init__(handlers=handlers, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "fake"

    @property
    def requests(self) -> list[FakeRequest]:
        return self._requests

    def _next_response(
        self,
        caller: str = "",
        schema_name: str | None = None,
        messages: list[Any] | None = None,
        request: Any = None,
    ) -> Any:
        key = (caller, schema_name) if (caller, schema_name) in self.handlers else caller
        if key not in self.handlers:
            raise KeyError(f"sem resposta roteirizada para {key!r}; disponíveis: {list(self.handlers)}")

        handler = self.handlers[key]
        indice = self._counters[key]
        self._counters[key] = indice + 1

        if isinstance(handler, list):
            return handler[min(indice, len(handler) - 1)]
        if callable(handler):
            return handler(request if request is not None else FakeRequest(messages or [], caller, schema_name))
        return handler

    @staticmethod
    def _as_message(item: Any) -> AIMessage:
        if isinstance(item, AIMessage):
            return item
        texto = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return AIMessage(
            content=texto,
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )

    @staticmethod
    def _caller_from_run(run_manager: Any) -> str:
        metadata = getattr(run_manager, "metadata", None) or {}
        if "caller" in metadata:
            return metadata["caller"]
        tags = getattr(run_manager, "tags", None) or []
        return next((tag for tag in tags if tag), "")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        caller = self._caller_from_run(run_manager)
        schema_name = kwargs.get("schema_name")
        message = self._as_message(self._next_response(caller, schema_name, messages))
        self._requests.append(FakeRequest(messages=list(messages), caller=caller, schema_name=schema_name))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        """Roteia pelo nome do schema e valida a saída como o LangChain faria."""

        def run(inp, config=None):
            message = self.invoke(inp, config=config, schema_name=schema.__name__)
            texto = message.content if isinstance(message.content, str) else json.dumps(message.content)
            try:
                parsed = schema.model_validate_json(texto)
                erro = None
            except (ValidationError, ValueError) as exc:
                parsed, erro = None, exc
            if include_raw:
                return {"raw": message, "parsed": parsed, "parsing_error": erro}
            if parsed is None:
                raise erro
            return parsed

        return RunnableLambda(run)

    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name: str, args: dict[str, Any], call_id: str = "call-1") -> AIMessage:
    """Resposta nativa de tool calling: o modelo pede a ferramenta `name`."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    )


def structured_final(schema_name: str, values: dict[str, Any], call_id: str = "call-final") -> AIMessage:
    """Resposta nativa de saída estruturada: o modelo chama a ferramenta sintética do schema."""
    return tool_call(schema_name, values, call_id)


class ScriptedLearner:
    def __init__(self, escolhas: dict[str, str] | None = None, padrao: str = "B") -> None:
        self._escolhas = escolhas or {}
        self._padrao = padrao
        self.turnos: list[TurnoTutor] = []
        self.itens: list[str] = []
        self.notificacoes: list[str] = []

    def reply_to_tutor(self, turno: TurnoTutor) -> str:
        self.turnos.append(turno)
        return "Acho que é assim..."

    def solve(self, item_id: str, item: ItemObjetivo) -> RespostaAluno:
        """`padrao`/`escolhas` aceitam uma letra ou "certa"/"errada" (as alternativas geradas são embaralhadas)."""
        self.itens.append(item_id)
        escolha = self._escolhas.get(item_id, self._padrao)
        if escolha == "certa":
            escolha = item.gabarito
        elif escolha == "errada":
            escolha = next(letra for letra in item.letras if letra != item.gabarito)
        return RespostaAluno(item_id=item_id, escolha=escolha, resolucao="fiz a conta")

    def notify(self, notificacao: Notificacao) -> None:
        self.notificacoes.append(notificacao.texto)


class RendererFalso:
    def __init__(self, falha: bool = False) -> None:
        self.falha, self.pedidos = falha, []

    def render(self, pedido: PedidoFigura) -> Figura:
        self.pedidos.append(pedido)
        if self.falha:
            raise FiguraError("código inválido")
        return Figura(
            descricao=pedido.descricao,
            formato=FormatoFigura.PNG,
            conteudo="iVBORw0KGgo=",
        )


FIGURA = {"tipo": "grafico", "codigo": "plt.plot([0, 1], [0, 1])", "descricao": "Reta identidade"}
