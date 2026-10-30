import time

from langchain_core.callbacks import BaseCallbackHandler

from estudai.application.telemetria import UsageLedger, UsageRecord


class UsageLedgerCallback(BaseCallbackHandler):
    """Callback to track LLM usage, token consumption and associated prices"""

    def __init__(self, ledger=None, *, model="fake", price_in=0.0, price_out=0.0):
        self._ledger = ledger or UsageLedger()
        self._model = model
        self._price_in = price_in
        self._price_out = price_out
        self._starts = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, tags=None, **kwargs):
        caller = (metadata or {}).get("caller") or next((t for t in (tags or []) if t), "desconhecido")
        self._starts[run_id] = (caller, time.perf_counter())

    def on_llm_end(self, response, *, run_id, **kwargs):
        caller, start = self._starts.pop(run_id, ("desconhecido", time.perf_counter()))
        message = response.generations[0][0].message
        usage = message.usage_metadata or {}
        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        self._ledger.record(
            UsageRecord(
                caller=caller,
                model=self._model,
                input_tokens=inp,
                output_tokens=out,
                latency_s=time.perf_counter() - start,
                cost_usd=(inp * self._price_in + out * self._price_out) / 1e6,
            )
        )
