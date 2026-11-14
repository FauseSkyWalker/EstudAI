"""Figuras que os agentes desenham durante a conversa (gráficos, diagramas).

O agente não produz a imagem: ele descreve o desenho em um `PedidoFigura` (código matplotlib ou
fonte Mermaid), que é renderizado na infraestrutura; o resultado (`Figura`) é anexado à mensagem
entregue ao aluno.
"""

import base64
from enum import StrEnum

from pydantic import BaseModel, Field


class TipoFigura(StrEnum):
    GRAFICO = "grafico"
    DIAGRAMA = "diagrama"


class FormatoFigura(StrEnum):
    PNG = "png"
    MERMAID = "mermaid"


class PedidoFigura(BaseModel):
    tipo: TipoFigura = Field(
        default=TipoFigura.GRAFICO,
        description=(
            "`grafico` para matplotlib (gráficos quantitativos e desenhos físicos) ou `diagrama` para "
            "Mermaid (fluxogramas, mapas mentais, árvores, linhas do tempo e diagramas de estados)."
        ),
    )
    codigo: str = Field(
        description=(
            "Para `grafico`: código Python curto com matplotlib (`plt` e `np` já estão importados; não chame "
            "plt.show()/plt.savefig() nem grave arquivos). Para `diagrama`: fonte Mermaid, ex.: "
            "'flowchart TD\\n  A --> B', sem links ou cliques. Títulos, rótulos e legendas em português; "
            "sem `$...$`/LaTeX, use matemática em texto simples e coloque rótulos com parênteses, "
            'colchetes, chaves, vírgulas ou sinais entre aspas duplas (ex.: `A["f(x) = ax + b"] --> B`).'
        ),
    )
    descricao: str = Field(
        min_length=1,
        description="Legenda curta do que a figura mostra. Não pode revelar a resposta da pergunta.",
    )


class Figura(BaseModel):
    descricao: str
    formato: FormatoFigura = FormatoFigura.PNG
    conteudo: str

    @property
    def png(self) -> bytes:
        if self.formato != FormatoFigura.PNG:
            raise ValueError("a figura não está no formato PNG")
        return base64.b64decode(self.conteudo)


class FiguraError(RuntimeError):
    """O pedido de figura falhou ou não produziu uma figura válida."""
