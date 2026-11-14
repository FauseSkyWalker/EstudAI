"""Run one complete study cycle and assemble its result."""

import time
from collections.abc import Callable
from functools import cached_property
from typing import Any

from estudai.application.cycle.deps import CycleDependencies
from estudai.application.cycle.result import ResultadoCiclo
from estudai.application.cycle.state import CycleState
from estudai.application.mensageria import MessageLog
from estudai.application.progresso import ObservadorProgresso, ProgressoCallback
from estudai.application.telemetria import UsageLedger
from estudai.domain.curriculum import Topico
from estudai.ports.learner import Learner

GraphFactory = Callable[[CycleDependencies], Any]


class StudyCycle:
    def __init__(
        self,
        deps: CycleDependencies,
        *,
        graph_factory: GraphFactory,
        ledger: UsageLedger | None = None,
    ) -> None:
        self._deps = deps
        self._graph_factory = graph_factory
        self._ledger = ledger or UsageLedger()

    @cached_property
    def _graph(self) -> Any:
        return self._graph_factory(self._deps)

    def executar(
        self,
        aluno_id: str,
        topico: Topico,
        aluno: Learner,
        observador: ObservadorProgresso | None = None,
    ) -> ResultadoCiclo:
        start = time.perf_counter()
        mark = self._ledger.mark()
        message_log = MessageLog()
        config = {"callbacks": [ProgressoCallback(observador)]} if observador else None
        state: CycleState = self._graph.invoke(
            {"student_id": aluno_id, "topic": topico, "learner": aluno, "log": message_log}, config=config
        )

        return ResultadoCiclo(
            aluno_id=aluno_id,
            topico=topico,
            adaptativo=self._deps.config.adaptativo,
            versao_perfil_inicial=state["initial_version"],
            versao_perfil_final=state["profile"].versao,
            sondagem=state.get("probe_ids", []),
            diagnostico_inicial=state.get("initial_diagnosis"),
            objetivo=state["objective"],
            material=state["material"],
            transcricao=state["transcript"],
            resumo=state.get("summary"),
            fornecimento=state["provision"],
            correcoes=state.get("corrections", []),
            removidos=state.get("removed_ids", []),
            diagnostico_final=state.get("final_diagnosis"),
            relatorio=state.get("report"),
            mensagens=message_log.mensagens,
            uso=self._ledger.summary(since=mark),
            latencia_s=time.perf_counter() - start,
        )
