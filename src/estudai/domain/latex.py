"""Notação matemática compartilhada: especificação para o LLM e conversão para o padrão do banco.

O LLM devolve matemática com `$...$` porque backslashes dentro de JSON/function calling são
frágeis. Convertemos para `\\(...\\)` depois do parse, que é o padrão do banco e dos agentes.
"""

import re

ESPECIFICACAO_LATEX = r"""
Notação matemática (LaTeX):
- Escreva TODA a matemática em LaTeX inline $...$, inclusive dentro dos campos do JSON.
- Use exclusivamente $...$; NÃO use \(...\), \[...\] ou $$...$$ no texto.
- Cifrões de moeda (R$) NÃO são matemática: preserve "R$ 5,00" literalmente, sem LaTeX.
- Use \frac, \sqrt, ^{}, _{}, \mathbb{R}, \le, \ge, \times, \pi, etc.
- Não use ^, sqrt(), x², "R" para os reais nem matemática em texto puro.
- Exemplo correto: "Se $f(x) = x^{2} + 1$, quanto vale $f(2)$?"
- Exemplo errado: "Se f(x) = x^2 + 1..." ou "Se \(f(x) = x² + 1\)...".
"""

_BLOCO = re.compile(r"\$\$(.+?)\$\$", re.S)
_COLCHETE = re.compile(r"\\\[(.+?)\\\]", re.S)
_INLINE = re.compile(r"(?<![A-Za-z0-9])\$(?!\s)([^$\n]+?)(?<!\s)\$(?![A-Za-z0-9])")


def para_inline(texto: str) -> str:
    """Converte `$...$`, `$$...$$` e `\\[...\\]` em `\\(...\\)`, preservando cifrões como `R$`."""
    texto = _BLOCO.sub(r"\\(\1\\)", texto)
    texto = _COLCHETE.sub(r"\\(\1\\)", texto)
    return _INLINE.sub(r"\\(\1\\)", texto)


def latex_valido(texto: str) -> bool:
    if texto.count(r"\(") != texto.count(r"\)") or r"\[" in texto or r"\]" in texto:
        return False
    return "$" not in texto.replace("R$", "").replace(r"\$", "")
