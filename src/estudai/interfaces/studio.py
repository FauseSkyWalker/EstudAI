"""Dev-only entry points for LangGraph Studio (`langgraph dev`).

Not imported by the CLI or web app. `ciclo_estudai` is view-only: the cycle state
carries `Learner`/`MessageLog` objects, which Studio cannot build from JSON.
`agente_conteudo` exposes the content agent's internal ReAct graph.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from estudai.agents.conteudo import AgenteConteudo
from estudai.application.cycle.deps import CycleDependencies
from estudai.bootstrap import Container
from estudai.orchestration.graph import build_cycle_graph
from estudai.settings import Settings


@lru_cache
def _container() -> Container:
    return Container(Settings())


def graph(deps: CycleDependencies | None = None) -> Any:
    return build_cycle_graph(deps or _container().orquestrador().deps)


def conteudo_graph(agente: AgenteConteudo | None = None) -> Any:
    return (agente or _container().conteudo).grafo_ferramentas()
