"""Public facade for the study-cycle pipeline."""

from collections.abc import Mapping

from estudai.agents.conteudo import AgenteConteudo
from estudai.agents.corretor import Corretor
from estudai.agents.diagnosticador import Diagnosticador
from estudai.agents.resumo import AgenteResumo
from estudai.agents.tutor import Tutor
from estudai.application.cycle.config import ConfigSessao
from estudai.application.cycle.deps import CycleDependencies, OnTutoring
from estudai.application.cycle.result import ResultadoCiclo
from estudai.application.cycle.runner import StudyCycle
from estudai.application.pratica import ExerciseProvider
from estudai.application.progresso import ObservadorProgresso
from estudai.application.telemetria import UsageLedger
from estudai.application.validacao import ValidadorExercicios
from estudai.domain.curriculum import Disciplina, Topico
from estudai.orchestration.graph import build_cycle_graph
from estudai.ports.figures import FigureRenderer
from estudai.ports.knowledge import QuestionBank
from estudai.ports.learner import Learner
from estudai.ports.repositories import ProfileRepository

__all__ = ["ConfigSessao", "Orquestrador", "ResultadoCiclo"]


class Orquestrador:
    def __init__(
        self,
        *,
        diagnosticador: Diagnosticador,
        tutores: Mapping[Disciplina, Tutor],
        conteudo: AgenteConteudo,
        resumo: AgenteResumo,
        corretor: Corretor,
        fornecedor: ExerciseProvider,
        validador: ValidadorExercicios,
        perfis: ProfileRepository,
        banco: QuestionBank,
        config: ConfigSessao | None = None,
        ledger: UsageLedger | None = None,
        figuras: FigureRenderer | None = None,
        max_figuras: int = 2,
        max_figuras_exercicio: int = 1,
        on_tutoring: OnTutoring | None = None,
    ) -> None:
        self._deps = CycleDependencies(
            diagnosticador=diagnosticador,
            tutores=tutores,
            conteudo=conteudo,
            resumo=resumo,
            corretor=corretor,
            fornecedor=fornecedor,
            validador=validador,
            perfis=perfis,
            banco=banco,
            config=config or ConfigSessao(),
            figuras=figuras,
            max_figuras=max_figuras,
            max_figuras_exercicio=max_figuras_exercicio,
            on_tutoring=on_tutoring,
        )
        self._tutores = tutores
        self._config = self._deps.config
        self._cycle = StudyCycle(self._deps, ledger=ledger, graph_factory=build_cycle_graph)

    @property
    def config(self) -> ConfigSessao:
        return self._config

    @property
    def deps(self) -> CycleDependencies:
        return self._deps

    def executar_ciclo(
        self,
        aluno_id: str,
        topico: Topico,
        aluno: Learner,
        observador: ObservadorProgresso | None = None,
    ) -> ResultadoCiclo:
        return self._cycle.executar(aluno_id, topico, aluno, observador)
