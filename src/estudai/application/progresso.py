"""Bastidores do pipeline: o que cada agente está fazendo, em tempo real.

Um callback do LangChain pendurado na execução do grafo recebe, por herança, os eventos de todas as
chamadas internas: início de cada etapa do grafo, cada chamada de modelo (com o agente que chamou,
via `metadata.caller`), cada ferramenta usada e eventos próprios (ex.: desenho de figura). Ele os
traduz em `EventoProgresso` legíveis e repassa a um observador (a interface web, o terminal...).
"""

import ast
import time
from collections.abc import Callable
from enum import StrEnum
from threading import Lock
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field

# Etapas do grafo (orchestration/graph.py), na ordem em que aparecem no ciclo
ETAPAS: dict[str, str] = {
    "load_profile": "Carregando o perfil",
    "diagnose": "Diagnóstico",
    "derive_objective": "Planejando a sessão",
    "retrieve_material": "Buscando material",
    "tutor": "Tutoria",
    "summarize": "Resumo",
    "provide_exercises": "Gerando exercícios",
    "practice": "Prática e correção",
    "update_profile": "Atualizando o perfil",
    "build_report": "Relatório",
    "record_episode": "Memória da sessão",
}

AGENTES: dict[str, str] = {
    "diagnosticador": "Diagnosticador",
    "tutor": "Tutor",
    "conteudo": "Agente de Conteúdo",
    "resumo": "Agente de Resumo",
    "gerador_exercicios": "Gerador de Exercícios",
    "painel_resolutores": "Painel de resolutores",
    "corretor": "Corretor",
}

EVENTO_FIGURA = "estudai_figura"  # nome do evento customizado emitido ao desenhar figuras


class TipoProgresso(StrEnum):
    ETAPA = "etapa"  # o grafo entrou numa etapa
    PENSANDO = "pensando"  # um agente chamou o modelo
    RESPONDEU = "respondeu"  # o modelo respondeu
    FERRAMENTA = "ferramenta"  # um agente usou uma ferramenta
    FIGURA = "figura"  # desenho de figura


class EventoProgresso(BaseModel):
    tipo: TipoProgresso
    descricao: str
    etapa: str | None = None  # chave de ETAPAS em andamento
    agente: str | None = None  # nome legível do agente
    instante: float = Field(default_factory=time.time)


ObservadorProgresso = Callable[[EventoProgresso], None]


def _nome_agente(metadata: dict | None) -> str | None:
    caller = (metadata or {}).get("caller")
    return AGENTES.get(caller, caller) if caller else None


def _entrada_ferramenta(input_str: str, inputs: dict | None) -> dict:
    if isinstance(inputs, dict):
        return inputs
    try:
        valor = ast.literal_eval(input_str)
    except (ValueError, SyntaxError):
        return {}
    return valor if isinstance(valor, dict) else {}


def descrever_ferramenta(nome: str, entrada: dict) -> str:
    if nome.startswith("busca_"):
        modo = nome.removeprefix("busca_")
        return f"busca {modo} na base: “{entrada.get('consulta', '')}”"
    if nome == "consultar_conteudo":
        return f"pede material ao Agente de Conteúdo: “{entrada.get('pedido', '')}”"
    if nome == "executar_python":
        return "confere as contas executando Python"
    return f"usa a ferramenta {nome}"


class ProgressoCallback(BaseCallbackHandler):
    """Traduz eventos do LangChain/LangGraph em `EventoProgresso` (thread-safe)."""

    raise_error = False  # um problema no observador nunca derruba o ciclo

    def __init__(self, observador: ObservadorProgresso) -> None:
        self._observador = observador
        self._etapa: str | None = None
        self._inicios: dict[Any, tuple[float, str | None]] = {}  # run_id -> (início, agente)
        self._lock = Lock()

    def _emitir(self, tipo: TipoProgresso, descricao: str, agente: str | None = None) -> None:
        self._observador(EventoProgresso(tipo=tipo, descricao=descricao, etapa=self._etapa, agente=agente))

    def on_chain_start(self, serialized, inputs, *, run_id, metadata=None, **kwargs) -> None:
        no = (metadata or {}).get("langgraph_node")
        # as chamadas internas herdam o `langgraph_node` da etapa; os nós do agente de ferramentas
        # ("model", "tools") não são etapas do ciclo e são ignorados
        if no in ETAPAS and kwargs.get("name") == no:
            with self._lock:
                mudou, self._etapa = no != self._etapa, no
            if mudou:
                self._emitir(TipoProgresso.ETAPA, ETAPAS[no])

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs) -> None:
        agente = _nome_agente(metadata)
        with self._lock:
            self._inicios[run_id] = (time.perf_counter(), agente)
        modelo = (metadata or {}).get("ls_model_name") or (kwargs.get("invocation_params") or {}).get("model")
        self._emitir(TipoProgresso.PENSANDO, "está pensando" + (f" ({modelo})" if modelo else ""), agente)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        with self._lock:
            registro = self._inicios.pop(run_id, None)
        if registro is not None:
            inicio, agente = registro
            self._emitir(TipoProgresso.RESPONDEU, f"respondeu em {time.perf_counter() - inicio:.1f}s", agente)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        with self._lock:
            self._inicios.pop(run_id, None)

    def on_tool_start(self, serialized, input_str, *, run_id, metadata=None, inputs=None, **kwargs) -> None:
        nome = (serialized or {}).get("name") or kwargs.get("name") or "ferramenta"
        descricao = descrever_ferramenta(nome, _entrada_ferramenta(input_str, inputs))
        self._emitir(TipoProgresso.FERRAMENTA, descricao, _nome_agente(metadata))

    def on_custom_event(self, name, data, *, run_id, metadata=None, **kwargs) -> None:
        if name == EVENTO_FIGURA and isinstance(data, dict):
            self._emitir(TipoProgresso.FIGURA, str(data.get("descricao", "")), data.get("agente"))
