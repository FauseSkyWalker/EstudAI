"""Dublês de teste: LLM roteirizado e aluno roteirizado."""

import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from estudai.domain.correcao import RespostaAluno
from estudai.domain.questoes import ItemObjetivo
from estudai.domain.tutoria import TurnoTutor
from estudai.ports.llm import LLMRequest, LLMResponse, Usage

Handler = Callable[[LLMRequest], str | dict[str, Any]]


class ScriptedLLM:
    """Responde por (caller, título do schema pedido). O handler pode ser uma função ou uma
    lista de respostas consumidas em ordem (a última se repete)."""

    def __init__(self, handlers: dict[str | tuple[str, str], Handler | list], model: str = "fake") -> None:
        self._handlers = handlers
        self._model = model
        self._contagem: dict[Any, int] = defaultdict(int)
        self.requests: list[LLMRequest] = []

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        titulo = (request.json_schema or {}).get("title", "")
        chave = (request.caller, titulo) if (request.caller, titulo) in self._handlers else request.caller
        handler = self._handlers[chave]
        if isinstance(handler, list):
            i = min(self._contagem[chave], len(handler) - 1)
            saida = handler[i]
        else:
            saida = handler(request)
        self._contagem[chave] += 1
        texto = saida if isinstance(saida, str) else json.dumps(saida, ensure_ascii=False)
        return LLMResponse(text=texto, model=self._model, usage=Usage(input_tokens=100, output_tokens=20))


def react_final(entrada: dict) -> dict:
    return {"pensamento": "pronto", "acao": "final", "entrada": entrada}


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
        self.itens.append(item_id)
        return RespostaAluno(
            item_id=item_id, escolha=self._escolhas.get(item_id, self._padrao), resolucao="fiz a conta"
        )

    def notify(self, mensagem: str) -> None:
        self.notificacoes.append(mensagem)
