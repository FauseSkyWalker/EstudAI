"""Tutoring summary phase."""

from estudai.agents.base import AgentOutputError
from estudai.application.cycle.deps import CycleDependencies
from estudai.application.mensageria import MessageLog
from estudai.domain.material import PacoteMaterial
from estudai.domain.mensagens import Papel, TipoMensagem
from estudai.domain.notificacao import Notificacao, TipoNotificacao
from estudai.domain.tutoria import ResumoEstudo, TranscricaoTutoria
from estudai.ports.learner import Learner


def summarize_tutoring(
    deps: CycleDependencies,
    transcricao: TranscricaoTutoria,
    material: PacoteMaterial,
    aluno: Learner,
    log: MessageLog,
) -> ResumoEstudo | None:
    if not transcricao.trocas:
        return None
    try:
        resumo = deps.resumo.resumir_tutoria(transcricao, material)
    except AgentOutputError:
        return None
    log.send(Papel.RESUMO, Papel.ALUNO, TipoMensagem.RESUMO, resumo)
    aluno.notify(Notificacao(tipo=TipoNotificacao.RESUMO, texto=format_summary(resumo)))
    return resumo


def format_summary(resumo: ResumoEstudo) -> str:
    partes = [f"# {resumo.titulo}", *[f"- {p}" for p in resumo.pontos_chave]]
    if resumo.formulas:
        partes += ["Fórmulas:", *[f"  {f}" for f in resumo.formulas]]
    return "\n".join(partes)
