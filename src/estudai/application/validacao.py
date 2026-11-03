"""Validação de itens gerados: verificação simbólica determinística + painel de resolutores.

Regra de aprovação (critério de sucesso 1, Seção 5.1):
- verificação simbólica DIVERGENTE  -> descarta;
- maioria do painel sinaliza mais de uma correta ou enunciado problemático -> descarta;
- verificação CONFIRMADA            -> aprova se concordância >= `min_concordancia`;
- verificação INCONCLUSIVA          -> aprova se concordância >= `min_concordancia_sem_verificacao`.
"""

import math
import re
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum

import sympy
from pydantic import BaseModel, Field
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from estudai.agents.base import AgentOutputError
from estudai.agents.resolutor import Resolutor, VotoResolutor
from estudai.domain.questoes import Exercicio, ItemObjetivo
from estudai.ports.tools import CodeExecutor

_NUMERO = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:/\d+(?:[.,]\d+)?)?")
_CARACTERES_SEGUROS = re.compile(r"[0-9A-Za-z\s.+\-*/()]*")
# a expressão vem de código gerado por LLM: só estes identificadores podem ser avaliados
_IDENTIFICADORES_PERMITIDOS = {"pi", "sqrt", "log", "ln", "exp", "E", "sin", "cos", "tan"}
# `parse_expr` usa eval internamente; restringir o `global_dict` evita qualquer nome do Python.
_GLOBALS = {
    "__builtins__": {},
    **{nome: getattr(sympy, nome) for nome in _IDENTIFICADORES_PERMITIDOS},
    "Integer": sympy.Integer,
    "Float": sympy.Float,
    "Rational": sympy.Rational,
    "Symbol": sympy.Symbol,
    "Function": sympy.Function,
}


class StatusVerificacao(StrEnum):
    CONFIRMADA = "confirmada"
    DIVERGENTE = "divergente"
    INCONCLUSIVA = "inconclusiva"


class ResultadoVerificacao(BaseModel):
    status: StatusVerificacao
    valor: float | None = None
    letras_compativeis: list[str] = Field(default_factory=list)
    detalhe: str = ""


class ResultadoValidacao(BaseModel):
    aprovado: bool
    motivo: str
    concordancia: float
    votos: list[VotoResolutor]
    verificacao: ResultadoVerificacao


def _para_float(expr: str) -> float | None:
    texto = expr.strip().replace(",", ".").replace("^", "**").replace("π", "pi").replace("√", "sqrt")
    if not _CARACTERES_SEGUROS.fullmatch(texto) or len(texto) > 200:
        return None
    if any(nome not in _IDENTIFICADORES_PERMITIDOS for nome in re.findall(r"[A-Za-z]+", texto)):
        return None
    try:
        expressao = parse_expr(texto, global_dict=_GLOBALS, transformations=standard_transformations)
        valor = complex(sympy.N(expressao))
    except (sympy.SympifyError, TypeError, ValueError, SyntaxError, AttributeError):
        return None
    return valor.real if abs(valor.imag) < 1e-12 else None


def valor_da_alternativa(texto: str) -> float | None:
    """Valor numérico de uma alternativa, se ela contiver exatamente um número."""
    numeros = _NUMERO.findall(texto.replace(" ", ""))
    return _para_float(numeros[0]) if len(numeros) == 1 else None


class VerificadorSimbolico:
    def __init__(self, executor: CodeExecutor, rel_tol: float = 1e-2) -> None:
        self._executor = executor
        self._rel_tol = rel_tol

    def verificar(self, item: Exercicio) -> ResultadoVerificacao:
        if not item.codigo_verificacao:
            return ResultadoVerificacao(status=StatusVerificacao.INCONCLUSIVA, detalhe="sem código")
        execucao = self._executor.run(item.codigo_verificacao)
        linhas = [ln for ln in execucao.stdout.splitlines() if ln.strip()]
        valor = _para_float(linhas[-1]) if execucao.ok and linhas else None
        if valor is None:
            return ResultadoVerificacao(
                status=StatusVerificacao.INCONCLUSIVA, detalhe=f"saída não numérica: {execucao.stderr[-200:]}"
            )
        valores = {letra: valor_da_alternativa(t) for letra, t in zip(item.letras, item.alternativas, strict=True)}
        compativeis = [
            letra
            for letra, v in valores.items()
            if v is not None and math.isclose(v, valor, rel_tol=self._rel_tol, abs_tol=1e-9)
        ]
        if len(compativeis) == 1:
            status = StatusVerificacao.CONFIRMADA if compativeis[0] == item.gabarito else StatusVerificacao.DIVERGENTE
        elif len(compativeis) > 1:
            status = StatusVerificacao.DIVERGENTE  # mais de uma alternativa com o valor correto
        elif all(v is not None for v in valores.values()):
            status = StatusVerificacao.DIVERGENTE  # todas numéricas e nenhuma bate
        else:
            status = StatusVerificacao.INCONCLUSIVA
        return ResultadoVerificacao(status=status, valor=valor, letras_compativeis=compativeis)


class PainelResolutores:
    def __init__(self, resolutores: list[Resolutor]) -> None:
        if not resolutores:
            raise ValueError("o painel precisa de ao menos um resolutor")
        self._resolutores = resolutores

    def votar(self, item: ItemObjetivo) -> list[VotoResolutor]:
        def resolver(r: Resolutor) -> VotoResolutor | None:
            try:
                return r.resolver(item)
            except AgentOutputError:
                return None

        with ThreadPoolExecutor(max_workers=len(self._resolutores)) as pool:
            votos = list(pool.map(resolver, self._resolutores))
        return [v for v in votos if v is not None]

    @property
    def tamanho(self) -> int:
        return len(self._resolutores)


class ValidadorExercicios:
    def __init__(
        self,
        painel: PainelResolutores,
        verificador: VerificadorSimbolico,
        *,
        min_concordancia: float = 2 / 3,
        min_concordancia_sem_verificacao: float = 1.0,
    ) -> None:
        self._painel = painel
        self._verificador = verificador
        self._min = min_concordancia
        self._min_sem_verificacao = min_concordancia_sem_verificacao

    def validar(self, item: Exercicio) -> ResultadoValidacao:
        verificacao = self._verificador.verificar(item)
        votos = self._painel.votar(item)
        n = self._painel.tamanho  # votos perdidos contam como discordância
        concordancia = sum(v.resposta == item.gabarito for v in votos) / n
        problematicos = sum(v.mais_de_uma_correta or v.enunciado_problematico for v in votos)

        def resultado(aprovado: bool, motivo: str) -> ResultadoValidacao:
            return ResultadoValidacao(
                aprovado=aprovado, motivo=motivo, concordancia=concordancia, votos=votos, verificacao=verificacao
            )

        if verificacao.status == StatusVerificacao.DIVERGENTE:
            return resultado(False, f"verificação simbólica divergente ({verificacao.letras_compativeis})")
        if problematicos > n / 2:
            return resultado(False, "painel sinalizou ambiguidade ou enunciado problemático")
        limiar = self._min if verificacao.status == StatusVerificacao.CONFIRMADA else self._min_sem_verificacao
        if concordancia + 1e-9 >= limiar:
            return resultado(True, f"consenso {concordancia:.2f} ({verificacao.status.value})")
        return resultado(False, f"sem consenso: {concordancia:.2f} < {limiar:.2f}")
