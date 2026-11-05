"""Converte o LaTeX armazenado no banco (`\\(...\\)`) para formato do streamlit (`$...$`)."""

import re

_MATH_INLINE = re.compile(r"\\\((.+?)\\\)", re.S)


def markdown_latex(texto: str) -> str:
    """Escapa `$` (preços como `R$`) e troca `\\(...\\)` por `$...$`."""
    return _MATH_INLINE.sub(lambda m: f"${m.group(1)}$", texto.replace("$", r"\$"))
