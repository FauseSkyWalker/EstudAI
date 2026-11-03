"""Shared state for the study-cycle graph.

Only cycle data lives here. Agents, repositories, the learner, telemetry, and the
message log stay outside the graph as dependencies.
"""

from typing import TypedDict

from estudai.application.pratica import ResultadoFornecimento
from estudai.domain.correcao import Correcao, RelatorioPratica
from estudai.domain.curriculum import Topico
from estudai.domain.material import PacoteMaterial
from estudai.domain.perfil import Diagnostico, PerfilAluno
from estudai.domain.planejamento import ObjetivoSessao
from estudai.domain.tutoria import ResumoEstudo, TranscricaoTutoria


class CycleState(TypedDict, total=False):
    student_id: str
    topic: Topico
    profile: PerfilAluno
    initial_version: int
    initial_diagnosis: Diagnostico | None
    probe_ids: list[str]
    objective: ObjetivoSessao
    material: PacoteMaterial
    transcript: TranscricaoTutoria
    summary: ResumoEstudo | None
    provision: ResultadoFornecimento
    corrections: list[Correcao]
    removed_ids: list[str]
    profile_before_update: PerfilAluno
    final_diagnosis: Diagnostico | None
    report: RelatorioPratica | None
