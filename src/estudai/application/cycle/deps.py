"""Dependencies shared by the phases of a study cycle."""

from collections.abc import Mapping
from dataclasses import dataclass

from estudai.agents.conteudo import AgenteConteudo
from estudai.agents.corretor import Corretor
from estudai.agents.diagnosticador import Diagnosticador
from estudai.agents.resumo import AgenteResumo
from estudai.agents.tutor import Tutor
from estudai.application.cycle.config import ConfigSessao
from estudai.application.pratica import ExerciseProvider
from estudai.application.validacao import ValidadorExercicios
from estudai.domain.curriculum import Disciplina
from estudai.ports.figures import FigureRenderer
from estudai.ports.knowledge import QuestionBank
from estudai.ports.repositories import ProfileRepository


@dataclass(frozen=True)
class CycleDependencies:
    diagnosticador: Diagnosticador
    tutores: Mapping[Disciplina, Tutor]
    conteudo: AgenteConteudo
    resumo: AgenteResumo
    corretor: Corretor
    fornecedor: ExerciseProvider
    validador: ValidadorExercicios
    perfis: ProfileRepository
    banco: QuestionBank
    config: ConfigSessao
    figuras: FigureRenderer | None = None
    max_figuras: int = 2
    max_figuras_exercicio: int = 1
