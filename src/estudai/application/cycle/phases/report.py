"""Practice report phase."""

from estudai.agents.base import AgentOutputError
from estudai.application.cycle.deps import CycleDependencies
from estudai.application.mensageria import MessageLog
from estudai.domain.correcao import Correcao, RelatorioPratica
from estudai.domain.curriculum import Topico
from estudai.domain.mensagens import Papel, TipoMensagem
from estudai.domain.notificacao import Notificacao, TipoNotificacao
from estudai.domain.perfil import PerfilAluno
from estudai.ports.learner import Learner


def build_report(
    deps: CycleDependencies,
    antes: PerfilAluno,
    depois: PerfilAluno,
    topico: Topico,
    correcoes: list[Correcao],
    aluno: Learner,
    log: MessageLog,
) -> RelatorioPratica | None:
    if not correcoes:
        return None
    try:
        relatorio = deps.resumo.resumir_pratica(correcoes, antes.visao(topico), depois.visao(topico))
    except AgentOutputError:
        return None
    log.send(Papel.RESUMO, Papel.ALUNO, TipoMensagem.RESUMO, relatorio)
    aluno.notify(Notificacao(tipo=TipoNotificacao.RELATORIO, texto=relatorio.desempenho))
    return relatorio
