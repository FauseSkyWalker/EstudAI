"""Figuras que os agentes desenham durante a conversa (gráficos, diagramas).

O agente não produz a imagem: ele descreve o desenho em código matplotlib (`PedidoFigura`), que é
executado na sandbox; o resultado (`Figura`) é anexado à mensagem entregue ao aluno.
"""

import base64

from pydantic import BaseModel, Field


class PedidoFigura(BaseModel):
    codigo: str = Field(
        description=(
            "Código Python que desenha UMA figura com matplotlib. `plt` (matplotlib.pyplot) e `np` (numpy) "
            "já estão importados. Não chame plt.show() nem plt.savefig(), não leia nem grave arquivos. "
            "Títulos, rótulos e legendas em português; fórmulas com mathtext, ex.: '$f(x)=x^2$'."
        )
    )
    descricao: str = Field(description="Legenda curta do que a figura mostra (também serve de texto alternativo).")


class Figura(BaseModel):
    descricao: str
    png_base64: str

    @property
    def png(self) -> bytes:
        return base64.b64decode(self.png_base64)


class FiguraError(RuntimeError):
    """O código da figura falhou ou não produziu imagem."""
