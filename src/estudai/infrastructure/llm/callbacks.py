"""Telemetria de uso (tokens, latência e custo por agente) via callbacks do LangChain."""

import time
from threading import Lock

from langchain_core.callbacks import BaseCallbackHandler

from estudai.application.telemetria import UsageLedger, UsageRecord


class UsageLedgerCallback(BaseCallbackHandler):
    """Registra cada chamada de LLM no `UsageLedger`.

    O agente é identificado por `metadata.caller` (ver `agents.base.run_config`). Os preços são
    os do modelo ao qual o callback está anexado, por isso há um callback por modelo/ledger.
    """

    def __init__(self, ledger: UsageLedger | None = None, *, model="fake", price_in=0.0, price_out=0.0):
        self._ledger = ledger or UsageLedger()
        self._model = model
        self._price_in = price_in
        self._price_out = price_out
        self._starts: dict = {}
        self._lock = Lock()  # o painel de resolutores chama o modelo em paralelo

    @property
    def ledger(self) -> UsageLedger:
        return self._ledger

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, tags=None, **kwargs):
        caller = (metadata or {}).get("caller") or next((t for t in (tags or []) if t), "desconhecido")
        with self._lock:
            self._starts[run_id] = (caller, time.perf_counter())

    def on_llm_end(self, response, *, run_id, **kwargs):
        with self._lock:
            caller, start = self._starts.pop(run_id, ("desconhecido", time.perf_counter()))
        message = response.generations[0][0].message
        usage = message.usage_metadata or {}
        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        self._ledger.record(
            UsageRecord(
                caller=caller,
                model=(response.llm_output or {}).get("model_name") or self._model,
                input_tokens=inp,
                output_tokens=out,
                latency_s=time.perf_counter() - start,
                cost_usd=self._custo(response, message, inp, out),
            )
        )

    def _custo(self, response, message, inp: int, out: int) -> float:
        informado = (getattr(message, "response_metadata", None) or {}).get("cost_usd")
        if informado is None:
            informado = ((response.llm_output or {}).get("token_usage") or {}).get("cost")
        return float(informado) if informado is not None else (inp * self._price_in + out * self._price_out) / 1e6

    def on_llm_error(self, error, *, run_id, **kwargs):
        # sem isto, chamadas que falham (após os retries do cliente) ficariam presas em `_starts`
        with self._lock:
            self._starts.pop(run_id, None)
