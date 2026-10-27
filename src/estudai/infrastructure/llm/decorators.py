"""Decoradores de `LLMClient`: retry com backoff e telemetria de custo/latência.

Compostos no bootstrap: Tracking(Retrying(adaptador)). Os agentes não sabem que existem.
"""

import time
from collections.abc import Callable

from estudai.application.telemetria import UsageLedger, UsageRecord
from estudai.ports.llm import LLMClient, LLMRequest, LLMResponse, TransientLLMError


class RetryingLLMClient:
    def __init__(
        self,
        inner: LLMClient,
        *,
        max_attempts: int = 4,
        base_delay_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._inner = inner
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._sleep = sleep

    @property
    def model(self) -> str:
        return self._inner.model

    def complete(self, request: LLMRequest) -> LLMResponse:
        for tentativa in range(1, self._max_attempts + 1):
            try:
                return self._inner.complete(request)
            except TransientLLMError:
                if tentativa == self._max_attempts:
                    raise
                self._sleep(self._base_delay_s * 2 ** (tentativa - 1))
        raise AssertionError("inalcançável")


class TrackingLLMClient:
    def __init__(
        self,
        inner: LLMClient,
        ledger: UsageLedger,
        *,
        price_input_per_mtok: float = 0.0,
        price_output_per_mtok: float = 0.0,
    ) -> None:
        self._inner = inner
        self._ledger = ledger
        self._price_in = price_input_per_mtok
        self._price_out = price_output_per_mtok

    @property
    def model(self) -> str:
        return self._inner.model

    def complete(self, request: LLMRequest) -> LLMResponse:
        inicio = time.perf_counter()
        response = self._inner.complete(request)
        uso = response.usage
        self._ledger.record(
            UsageRecord(
                caller=request.caller,
                model=response.model,
                input_tokens=uso.input_tokens,
                output_tokens=uso.output_tokens,
                latency_s=time.perf_counter() - inicio,
                cost_usd=(uso.input_tokens * self._price_in + uso.output_tokens * self._price_out) / 1e6,
            )
        )
        return response
