"""Conversão de delimitadores de matemática para o padrão do banco (`\\(...\\)`).

O LLM devolve matemática com `$...$` porque backslashes dentro de JSON/function calling são
frágeis (o modelo esquece de escapar `\\(`). Convertemos para `\\(...\\)` só depois do parse,
que é o formato usado pelos agentes, pelo índice e pela interface.
"""

import re

_BLOCO = re.compile(r"\$\$(.+?)\$\$", re.S)
_COLCHETE = re.compile(r"\\\[(.+?)\\\]", re.S)
_INLINE = re.compile(r"(?<![A-Za-z0-9])\$(?!\s)([^$\n]+?)(?<!\s)\$(?![A-Za-z0-9])")


def para_inline(texto: str) -> str:
    """Converte `$...$`, `$$...$$` e `\\[...\\]` em `\\(...\\)`, preservando cifrões como `R$`."""
    texto = _BLOCO.sub(r"\\(\1\\)", texto)
    texto = _COLCHETE.sub(r"\\(\1\\)", texto)
    return _INLINE.sub(r"\\(\1\\)", texto)


def latex_valido(texto: str) -> bool:
    """Delimitadores `\\(...\\)` balanceados e nenhum `$` fora de moeda (R$)."""
    if texto.count(r"\(") != texto.count(r"\)") or r"\[" in texto or r"\]" in texto:
        return False
    return "$" not in texto.replace("R$", "").replace(r"\$", "")
