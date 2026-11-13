from typing import Protocol

from pydantic import BaseModel


class ExecutionResult(BaseModel):
    ok: bool
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class CodeExecutor(Protocol):
    """Executor Python isolado: determinístico, sem rede e com limite de tempo (Seção 4)."""

    def run(self, code: str, timeout_s: float | None = None) -> ExecutionResult: ...
