"""Normalização estrutural das bases públicas de questões, feito como rodada de robustez prévia.

As bases de questões utilizadas originalmente utilizam MathML, HTML e marcadores de imagens.
Realizamos uma rodada inicial de normalização e limpeza de texto determinística para maior robustez
na chamada de LLM.

As regras abaixo são heurísticas feitas com uso de LLMs
"""

import html
import re

from mathml_to_latex.converter import MathMLToLaTeX

_FALHA_CONVERSAO = "\ufffd"

_MATH = re.compile(r"<(?:[a-zA-Z0-9]+:)?math\b[^>]*>.*?</(?:[a-zA-Z0-9]+:)?math>", re.S | re.I)
_MATH_ABERTA = re.compile(r"<(?:[a-zA-Z0-9]+:)?math\b", re.I)
_TAGS = re.compile(
    r"</?(?:math|mi|mn|mo|mfrac|msup|msub|msubsup|mfenced|mrow|mtext|mspace|msqrt|mroot|"
    r"mtable|mtr|mtd|mover|munder|munderover|mstyle|menclose|maction|mpadded|mphantom|"
    r"br|p|span|sup|sub|b|i|u|em|strong|img)\b[^>]*>",
    re.I,
)
_PLACEHOLDER = re.compile(r"\[\[placeholder\]\]|\[IMAGE \d+\]", re.I)
_ENTIDADE = re.compile(r"&(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);")
_ESPACOS = re.compile(r"[ \t]+")
_FUNCOES = re.compile(r"(?<!\\)\b(cos|sin|tan|log|ln|exp|lim|max|min)\b")

# Referências a figuras/tabelas que a base ENEM Challenge não traz (não há coluna de imagem).
# Perguntas assim são insolúveis e precisam ser descartadas na ingestão.
_REFERENCIA_FIGURA = re.compile(
    r"(?i)(?:"
    r"\b(?:na|nessa|nesta|conforme\s+a|de\s+acordo\s+com\s+a|observe\s+a|veja\s+a)\s+"
    r"(?:figura|imagem|tabela|gr[aá]fico|charge|tirinha|quadrinho|esquema|diagrama|mapa)\b"
    r"|\b(?:figura|imagem|tabela|gr[aá]fico|charge|tirinha|quadrinho|esquema|diagrama|mapa)\s+"
    r"(?:a\s+seguir|abaixo|acima|apresentad[ao]s?|mostrad[ao]s?)\b"
    r"|\ba\s+figura\s+(?:ilustra|exemplifica|mostra|representa|apresenta)\b"
    r"|\bo\s+gr[aá]fico\s+(?:mostra|representa|indica|ilustra)\b"
    r"|\ba\s+tabela\s+(?:mostra|apresenta|indica)\b"
    r"|\bno\s+gr[aá]fico\b"
    r"|\b(?:charge|tirinha|quadrinho)\b"
    r"|\b(?:esquema|diagrama)\b"
    r")"
)


def _math_para_latex(match: re.Match) -> str:
    try:
        latex = MathMLToLaTeX.convert(match.group(0)).strip()
    except Exception:  # noqa: BLE001
        return _FALHA_CONVERSAO
    if not latex:
        return _FALHA_CONVERSAO
    latex = _FUNCOES.sub(r"\\\1", latex)
    return f"\\({latex}\\)"


def limpar_texto(texto: str) -> str:
    """Converte MathML em LaTeX e remove marcação HTML/XML residual."""
    if not texto:
        return ""
    texto = _MATH.sub(_math_para_latex, texto)
    if _MATH_ABERTA.search(texto):  # MathML truncado: não dá para converter
        texto = _MATH_ABERTA.sub(_FALHA_CONVERSAO, texto)
    texto = _TAGS.sub(" ", texto)
    return _ESPACOS.sub(" ", html.unescape(texto)).strip()


def referencia_figura(texto: str) -> bool:
    """Indica se o enunciado depende de uma figura/tabela que não é presente."""
    return bool(_REFERENCIA_FIGURA.search(texto))


def texto_invalido(texto: str) -> bool:
    """Sobrou marcação, entidade, placeholder ou falha de conversão de MathML?"""
    return _FALHA_CONVERSAO in texto or bool(
        _PLACEHOLDER.search(texto) or _MATH_ABERTA.search(texto) or _TAGS.search(texto) or _ENTIDADE.search(texto)
    )
