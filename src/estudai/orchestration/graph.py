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
    from estudai.application.orquestrador import Orquestrador


def build_cycle_graph(orchestrator: Orquestrador) -> Any:
    def load_profile(state: CycleState) -> dict[str, Any]:
        profile = orchestrator._load_profile(state["student_id"])
        return {"profile": profile, "initial_version": profile.versao}

    def diagnose(state: CycleState) -> dict[str, Any]:
        profile, diagnosis, probe_ids = orchestrator._run_initial_diagnosis(
            state["profile"], state["topic"], state["learner"], state["log"]
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
            state["log"],
        )
        return {"material": material}

    def tutor(state: CycleState) -> dict[str, Any]:
        transcript, material = orchestrator._run_tutoring(
            state["profile"],
            state["objective"],
            state["material"],
            state["learner"],
            state["log"],
        )
        return {"transcript": transcript, "material": material}

    def summarize(state: CycleState) -> dict[str, Any]:
        summary = orchestrator._summarize_tutoring(
            state["transcript"],
            state["material"],
            state["learner"],
            state["log"],
        )
        return {"summary": summary}

    def provide_exercises(state: CycleState) -> dict[str, Any]:
        provision = orchestrator._provide_exercises(
            state["profile"],
            state["topic"],
            frozenset(state.get("probe_ids", [])),
            state["log"],
        )
        return {"provision": provision}

    def practice(state: CycleState) -> dict[str, Any]:
        corrections, removed_ids = orchestrator._run_practice(
            state["provision"].exercicios,
            state["learner"],
            state["log"],
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
            state["log"],
        )
        return {"profile": profile, "final_diagnosis": diagnosis}

    def build_report(state: CycleState) -> dict[str, Any]:
        before = state.get("profile_before_update", state["profile"])
        report = orchestrator._build_report(
            before,
            state["profile"],
            state["topic"],
            state["corrections"],
            state["learner"],
            state["log"],
        )
        return {"report": report}

    def record_episode(state: CycleState) -> dict[str, Any]:
        profile = orchestrator._record_episode(
            state["profile"],
            state["objective"],
            state.get("transcript"),
            state.get("summary"),
            state.get("report"),
            state.get("corrections", []),
            state["log"],
        )
        return {"profile": profile}

    def needs_diagnosis(state: CycleState) -> str:
        if orchestrator.config.adaptativo and not state["profile"].diagnosticado(state["topic"]):
            return "diagnose"
        return "derive_objective"

    def needs_profile_update(_state: CycleState) -> str:
        return "update_profile" if orchestrator.config.adaptativo else "build_report"

    def needs_episode(_state: CycleState) -> str:
        # C2 (perfil congelado) também congela a memória episódica
        return "record_episode" if orchestrator.config.adaptativo else END

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
    graph.add_node("record_episode", record_episode)

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
    graph.add_conditional_edges("build_report", needs_episode, {"record_episode": "record_episode", END: END})
    graph.add_edge("record_episode", END)

    return graph.compile()
