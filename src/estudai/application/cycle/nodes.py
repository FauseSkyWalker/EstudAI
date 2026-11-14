"""State adapters that expose cycle phases as LangGraph nodes."""

from collections.abc import Callable
from typing import Any

from estudai.application.cycle.deps import CycleDependencies
from estudai.application.cycle.phases import (
    diagnosis,
    exercises,
    material,
    planning,
    profile,
    report,
    summary,
    tutoring,
)
from estudai.application.cycle.phases import (
    practice as practice_phases,
)
from estudai.application.cycle.state import CycleState

Node = Callable[[CycleState], dict[str, Any]]


def load_profile(deps: CycleDependencies, state: CycleState) -> dict[str, Any]:
    perfil, versao = profile.load_profile(deps, state["student_id"])
    return {"profile": perfil, "initial_version": versao}


def diagnose(deps: CycleDependencies, state: CycleState) -> dict[str, Any]:
    perfil, diagnostico, probe_ids = diagnosis.run_initial_diagnosis(
        deps, state["profile"], state["topic"], state["learner"], state["log"]
    )
    return {"profile": perfil, "initial_diagnosis": diagnostico, "probe_ids": probe_ids}


def derive_objective(deps: CycleDependencies, state: CycleState) -> dict[str, Any]:
    objetivo = planning.derive_objective(deps, state["profile"], state["topic"])
    return {"objective": objetivo}


def retrieve_material(deps: CycleDependencies, state: CycleState) -> dict[str, Any]:
    material_ = material.retrieve_material(deps, state["profile"], state["objective"], state["log"])
    return {"material": material_}


def tutor(deps: CycleDependencies, state: CycleState) -> dict[str, Any]:
    transcricao, material_ = tutoring.run_tutoring(
        deps, state["profile"], state["objective"], state["material"], state["learner"], state["log"]
    )
    return {"transcript": transcricao, "material": material_}


def summarize(deps: CycleDependencies, state: CycleState) -> dict[str, Any]:
    resumo = summary.summarize_tutoring(deps, state["transcript"], state["material"], state["learner"], state["log"])
    return {"summary": resumo}


def provide_exercises(deps: CycleDependencies, state: CycleState) -> dict[str, Any]:
    fornecimento = exercises.provide_exercises(
        deps, state["profile"], state["topic"], frozenset(state.get("probe_ids", [])), state["log"]
    )
    return {"provision": fornecimento}


def practice(deps: CycleDependencies, state: CycleState) -> dict[str, Any]:
    correcoes, removidos = practice_phases.run_practice(
        deps, state["provision"].exercicios, state["learner"], state["log"]
    )
    return {
        "corrections": correcoes,
        "removed_ids": removidos,
        "profile_before_update": state["profile"],
    }


def update_profile(deps: CycleDependencies, state: CycleState) -> dict[str, Any]:
    perfil, diagnostico = profile.update_profile(
        deps, state["profile"], state["topic"], state["provision"], state["corrections"], state["log"]
    )
    return {"profile": perfil, "final_diagnosis": diagnostico}


def build_report(deps: CycleDependencies, state: CycleState) -> dict[str, Any]:
    antes = state.get("profile_before_update", state["profile"])
    relatorio = report.build_report(
        deps, antes, state["profile"], state["topic"], state["corrections"], state["learner"], state["log"]
    )
    return {"report": relatorio}


def record_episode(deps: CycleDependencies, state: CycleState) -> dict[str, Any]:
    perfil = profile.record_episode(
        deps,
        state["profile"],
        state["objective"],
        state.get("transcript"),
        state.get("summary"),
        state.get("report"),
        state.get("corrections", []),
        state["log"],
    )
    return {"profile": perfil}


def needs_diagnosis(deps: CycleDependencies, state: CycleState) -> str:
    if deps.config.adaptativo and not state["profile"].diagnosticado(state["topic"]):
        return "diagnose"
    return "derive_objective"


def needs_profile_update(deps: CycleDependencies, _state: CycleState) -> str:
    return "update_profile" if deps.config.adaptativo else "build_report"


def needs_episode(deps: CycleDependencies, _state: CycleState) -> str:
    return "record_episode" if deps.config.adaptativo else "end"


NODE_FUNCTIONS: dict[str, Callable[[CycleDependencies, CycleState], dict[str, Any]]] = {
    "load_profile": load_profile,
    "diagnose": diagnose,
    "derive_objective": derive_objective,
    "retrieve_material": retrieve_material,
    "tutor": tutor,
    "summarize": summarize,
    "provide_exercises": provide_exercises,
    "practice": practice,
    "update_profile": update_profile,
    "build_report": build_report,
    "record_episode": record_episode,
}


def make_nodes(deps: CycleDependencies) -> dict[str, Node]:
    def bind(funcao: Callable[[CycleDependencies, CycleState], dict[str, Any]]) -> Node:
        def node(state: CycleState) -> dict[str, Any]:
            return funcao(deps, state)

        node.__name__ = funcao.__name__
        node.__qualname__ = funcao.__qualname__
        node.__doc__ = funcao.__doc__
        return node

    return {nome: bind(funcao) for nome, funcao in NODE_FUNCTIONS.items()}
