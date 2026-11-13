from typing import Protocol

from estudai.domain.figura import Figura, PedidoFigura


class FigureRenderer(Protocol):
    """Renderiza o pedido de figura de um agente (matplotlib ou Mermaid).

    Levanta `estudai.domain.figura.FiguraError` se o pedido falhar ou não produzir figura válida.
    """

    def render(self, pedido: PedidoFigura) -> Figura: ...
