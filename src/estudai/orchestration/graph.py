"""Thin LangGraph wrapper around the study-cycle phases."""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from estudai.application.cycle.deps import CycleDependencies
from estudai.application.cycle.nodes import (
    make_nodes,
    needs_diagnosis,
    needs_episode,
    needs_profile_update,
)
from estudai.application.cycle.state import CycleState


def build_cycle_graph(deps: CycleDependencies) -> Any:
    graph = StateGraph(CycleState)

    for nome, node in make_nodes(deps).items():
        graph.add_node(nome, node)

    graph.add_edge(START, "load_profile")
    graph.add_conditional_edges(
        "load_profile",
        partial(needs_diagnosis, deps),
        {"diagnose": "diagnose", "derive_objective": "derive_objective"},
    )
    graph.add_edge("diagnose", "derive_objective")
    graph.add_edge("derive_objective", "retrieve_material")
    graph.add_edge("retrieve_material", "tutor")
    graph.add_edge("tutor", "summarize")
    graph.add_edge("summarize", "provide_exercises")
    graph.add_edge("provide_exercises", "practice")
    graph.add_conditional_edges(
        "practice",
        partial(needs_profile_update, deps),
        {"update_profile": "update_profile", "build_report": "build_report"},
    )
    graph.add_edge("update_profile", "build_report")
    graph.add_conditional_edges(
        "build_report",
        partial(needs_episode, deps),
        {"record_episode": "record_episode", "end": END},
    )
    graph.add_edge("record_episode", END)

    return graph.compile()
