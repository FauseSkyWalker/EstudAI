from typing import Protocol

from pydantic import BaseModel

from estudai.domain.figura import Figura, PedidoFigura


class ExecutionResult(BaseModel):
    ok: bool
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class CodeExecutor(Protocol):
    """Executor Python isolado: determinístico, sem rede e com limite de tempo (Seção 4)."""

    def run(self, code: str, timeout_s: float | None = None) -> ExecutionResult: ...


class FigureRenderer(Protocol):
    """Desenha a figura descrita por um agente (código matplotlib) num ambiente isolado.

    Levanta `estudai.domain.figura.FiguraError` se o código falhar ou não produzir imagem.
    """

    def render(self, pedido: PedidoFigura) -> Figura: ...
