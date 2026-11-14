"""Study-cycle configuration, dependencies, result, and runner."""

from estudai.application.cycle.config import ConfigSessao
from estudai.application.cycle.deps import CycleDependencies
from estudai.application.cycle.result import ResultadoCiclo

__all__ = ["ConfigSessao", "CycleDependencies", "ResultadoCiclo", "StudyCycle"]


def __getattr__(name: str):
    if name == "StudyCycle":
        from estudai.application.cycle.runner import StudyCycle

        return StudyCycle
    raise AttributeError(name)
