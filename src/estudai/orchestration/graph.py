"""Thin LangGraph wrapper around the existing study-cycle phases.

The graph controls the order of the phases. Agent behavior, persistence, telemetry,
and learner interaction remain in the orchestrator and its collaborators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from estudai.domain.planejamento import ObjetivoSessao
from estudai.orchestration.state import CycleState

if TYPE_CHECKING:
    from estudai.application.mensageria import MessageLog
    from estudai.application.orquestrador import Orquestrador
    from estudai.ports.learner import Learner


def build_cycle_graph(orchestrator: Orquestrador, learner: Learner, message_log: MessageLog) -> Any:
    def load_profile(state: CycleState) -> dict[str, Any]:
        profile = orchestrator._load_profile(state["student_id"])
        return {"profile": profile, "initial_version": profile.versao}

    def diagnose(state: CycleState) -> dict[str, Any]:
        profile, diagnosis, probe_ids = orchestrator._run_initial_diagnosis(
            state["profile"], state["topic"], learner, message_log
        )
        return {
            "profile": profile,
            "initial_diagnosis": diagnosis,
            "probe_ids": probe_ids,
        }

    def derive_objective(state: CycleState) -> dict[str, Any]:
        objective = ObjetivoSessao.derivar(
            state["profile"],
            state["topic"],
            orchestrator.config.conceitos_por_sessao,
        )
        return {"objective": objective}

    def retrieve_material(state: CycleState) -> dict[str, Any]:
        material = orchestrator._retrieve_material(
            state["profile"],
            state["objective"],
            message_log,
        )
        return {"material": material}

    def tutor(state: CycleState) -> dict[str, Any]:
        transcript = orchestrator._run_tutoring(
            state["profile"],
            state["objective"],
            state["material"],
            learner,
            message_log,
        )
        return {"transcript": transcript}

    def summarize(state: CycleState) -> dict[str, Any]:
        summary = orchestrator._summarize_tutoring(
            state["transcript"],
            state["material"],
            learner,
            message_log,
        )
        return {"summary": summary}

    def provide_exercises(state: CycleState) -> dict[str, Any]:
        provision = orchestrator._provide_exercises(
            state["profile"],
            state["topic"],
            frozenset(state.get("probe_ids", [])),
            message_log,
        )
        return {"provision": provision}

    def practice(state: CycleState) -> dict[str, Any]:
        corrections, removed_ids = orchestrator._run_practice(
            state["provision"].exercicios,
            learner,
            message_log,
        )
        return {
            "corrections": corrections,
            "removed_ids": removed_ids,
            "profile_before_update": state["profile"],
        }

    def update_profile(state: CycleState) -> dict[str, Any]:
        profile, diagnosis = orchestrator._update_profile(
            state["profile"],
            state["topic"],
            state["provision"],
            state["corrections"],
            message_log,
        )
        return {"profile": profile, "final_diagnosis": diagnosis}

    def build_report(state: CycleState) -> dict[str, Any]:
        before = state.get("profile_before_update", state["profile"])
        report = orchestrator._build_report(
            before,
            state["profile"],
            state["topic"],
            state["corrections"],
            learner,
            message_log,
        )
        return {"report": report}

    def needs_diagnosis(state: CycleState) -> str:
        if orchestrator.config.adaptativo and not state["profile"].diagnosticado(state["topic"]):
            return "diagnose"
        return "derive_objective"

    def needs_profile_update(_state: CycleState) -> str:
        return "update_profile" if orchestrator.config.adaptativo else "build_report"

    graph = StateGraph(CycleState)

    graph.add_node("load_profile", load_profile)
    graph.add_node("diagnose", diagnose)
    graph.add_node("derive_objective", derive_objective)
    graph.add_node("retrieve_material", retrieve_material)
    graph.add_node("tutor", tutor)
    graph.add_node("summarize", summarize)
    graph.add_node("provide_exercises", provide_exercises)
    graph.add_node("practice", practice)
    graph.add_node("update_profile", update_profile)
    graph.add_node("build_report", build_report)

    graph.add_edge(START, "load_profile")
    graph.add_conditional_edges(
        "load_profile",
        needs_diagnosis,
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
        needs_profile_update,
        {"update_profile": "update_profile", "build_report": "build_report"},
    )
    graph.add_edge("update_profile", "build_report")
    graph.add_edge("build_report", END)

    return graph.compile()
