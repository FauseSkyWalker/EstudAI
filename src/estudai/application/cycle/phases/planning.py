"""Session objective derivation."""

from estudai.application.cycle.deps import CycleDependencies
from estudai.domain.curriculum import Topico
from estudai.domain.perfil import PerfilAluno
from estudai.domain.planejamento import ObjetivoSessao


def derive_objective(deps: CycleDependencies, perfil: PerfilAluno, topico: Topico) -> ObjetivoSessao:
    return ObjetivoSessao.derivar(perfil, topico, deps.config.conceitos_por_sessao)
