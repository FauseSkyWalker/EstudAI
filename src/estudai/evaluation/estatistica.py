"""Análise estatística da Seção 5.3: Wilcoxon pareado, IC por bootstrap e κ de Cohen/Fleiss."""

from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats


def wilcoxon_pareado(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    diferencas = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    resultado = {
        "n": float(len(diferencas)),
        "mediana_diferenca": float(np.median(diferencas)) if len(diferencas) else 0.0,
    }
    if len(diferencas) == 0 or np.allclose(diferencas, 0):
        return resultado | {"estatistica": 0.0, "p_valor": 1.0}
    teste = stats.wilcoxon(diferencas, zero_method="zsplit")
    return resultado | {"estatistica": float(teste.statistic), "p_valor": float(teste.pvalue)}


def bootstrap_ic(
    valores: Sequence[float],
    *,
    estatistica: Callable[[np.ndarray], float] = np.mean,
    reamostras: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    x = np.asarray(valores, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    amostras = rng.choice(x, size=(reamostras, len(x)), replace=True)
    valores_boot = np.apply_along_axis(estatistica, 1, amostras)
    return float(np.quantile(valores_boot, alpha / 2)), float(np.quantile(valores_boot, 1 - alpha / 2))


def cohen_kappa(r1: Sequence, r2: Sequence) -> float:
    if len(r1) != len(r2) or not r1:
        raise ValueError("avaliações pareadas de mesmo tamanho são necessárias")
    n = len(r1)
    observado = sum(a == b for a, b in zip(r1, r2, strict=True)) / n
    c1, c2 = Counter(r1), Counter(r2)
    esperado = sum(c1[k] * c2[k] for k in set(c1) | set(c2)) / n**2
    return 1.0 if esperado == 1 else (observado - esperado) / (1 - esperado)


def fleiss_kappa(avaliacoes: Sequence[Sequence]) -> float:
    """`avaliacoes[i]` = rótulos dados ao item i pelos avaliadores (mesmo nº por item)."""
    itens = [list(a) for a in avaliacoes if len(a) >= 2]
    if not itens:
        return float("nan")
    m = len(itens[0])
    if any(len(a) != m for a in itens):
        raise ValueError("todos os itens precisam do mesmo número de avaliadores")
    categorias = sorted({c for a in itens for c in a}, key=str)
    contagens = np.array([[Counter(a)[c] for c in categorias] for a in itens], dtype=float)
    n_itens = len(itens)
    p_j = contagens.sum(axis=0) / (n_itens * m)
    p_i = ((contagens**2).sum(axis=1) - m) / (m * (m - 1))
    p_barra, p_e = p_i.mean(), (p_j**2).sum()
    return 1.0 if p_e == 1 else float((p_barra - p_e) / (1 - p_e))
