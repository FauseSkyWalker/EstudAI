"""Configuration for one study cycle."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigSessao:
    adaptativo: bool = True
    max_rodadas_tutoria: int = 6
    exercicios_por_sessao: int = 8
    conceitos_por_sessao: int = 2
    seed: int = 0
