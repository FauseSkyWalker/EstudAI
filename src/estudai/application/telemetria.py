"""Registro de uso de LLM (tokens, latência e custo) por agente: métricas da Seção 5.2.3."""

from collections import defaultdict
from threading import Lock

from pydantic import BaseModel, Field


class UsageRecord(BaseModel):
    caller: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float


class UsageTotals(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0

    def add(self, r: UsageRecord) -> None:
        self.calls += 1
        self.input_tokens += r.input_tokens
        self.output_tokens += r.output_tokens
        self.latency_s += r.latency_s
        self.cost_usd += r.cost_usd


class UsageSummary(BaseModel):
    total: UsageTotals = Field(default_factory=UsageTotals)
    por_agente: dict[str, UsageTotals] = Field(default_factory=dict)


class UsageLedger:
    """Livro-razão thread-safe (o painel de resolutores chama modelos em paralelo)."""

    def __init__(self) -> None:
        self._records: list[UsageRecord] = []
        self._lock = Lock()

    def record(self, record: UsageRecord) -> None:
        with self._lock:
            self._records.append(record)

    def mark(self) -> int:
        with self._lock:
            return len(self._records)

    def summary(self, since: int = 0) -> UsageSummary:
        with self._lock:
            records = self._records[since:]
        por_agente: dict[str, UsageTotals] = defaultdict(UsageTotals)
        total = UsageTotals()
        for r in records:
            total.add(r)
            por_agente[r.caller].add(r)
        return UsageSummary(total=total, por_agente=dict(por_agente))
