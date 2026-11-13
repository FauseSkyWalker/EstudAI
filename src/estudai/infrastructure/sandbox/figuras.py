"""Desenha as figuras pedidas pelos agentes: código matplotlib executado na sandbox.

O matplotlib é carregado antes do bloqueio de escrita (ele grava o cache de fontes ao carregar);
o código do agente roda depois, sem rede e com escrita bloqueada pelo audit hook. A imagem volta
pela saída padrão, em base64. O PNG nunca é gravado em disco por este renderer.
"""

import base64
import binascii
import os
import tempfile

from estudai.domain.figura import Figura, FiguraError, PedidoFigura
from estudai.infrastructure.sandbox.python_executor import SubprocessPythonExecutor

_MARCADOR = "<<ESTUDAI_PNG>>"
_PNG = b"\x89PNG\r\n\x1a\n"

_PRELUDIO_MPL = """
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
"""

_SALVAR = f"""
import io as _io, base64 as _b64
if not plt.get_fignums():
    raise RuntimeError("o código não desenhou nenhuma figura")
_buf = _io.BytesIO()
plt.gcf().savefig(_buf, format="png", dpi=110, bbox_inches="tight")
print("{_MARCADOR}" + _b64.b64encode(_buf.getvalue()).decode())
"""


class MatplotlibRenderer:
    def __init__(self, *, timeout_s: float = 30.0, max_bytes: int = 2_000_000) -> None:
        self._max_bytes = max_bytes
        self._executor = SubprocessPythonExecutor(
            timeout_s=timeout_s,
            preludio_confiavel=_PRELUDIO_MPL,
            # cache de fontes persistente
            env_extra={"MPLCONFIGDIR": os.path.join(tempfile.gettempdir(), "estudai-mplconfig"), "MPLBACKEND": "Agg"},
            max_saida=4 * max_bytes + 1_000,  # base64 ocupa ~4/3 do PNG
            importar_sympy=False,
        )

    def render(self, pedido: PedidoFigura) -> Figura:
        resultado = self._executor.run(pedido.codigo + "\n" + _SALVAR)
        if resultado.timed_out:
            raise FiguraError("tempo limite excedido ao desenhar a figura")
        # o marcador do próprio renderer é a última linha; saídas do agente podem vir antes
        linha = next(
            (li for li in reversed(resultado.stdout.splitlines()) if li.startswith(_MARCADOR)),
            None,
        )
        if not resultado.ok or linha is None:
            raise FiguraError(f"o código da figura falhou: {resultado.stderr.strip()[-400:]}")

        codificado = linha[len(_MARCADOR) :].strip()
        if len(codificado) > 4 * self._max_bytes // 3 + 4:
            raise FiguraError("a saída da figura é grande demais")
        try:
            png = base64.b64decode(codificado, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise FiguraError("a saída da figura não é base64 válido") from exc
        if not png.startswith(_PNG) or len(png) > self._max_bytes:
            raise FiguraError("a saída não é um PNG válido ou é grande demais")
        return Figura(descricao=pedido.descricao, png_base64=codificado)
