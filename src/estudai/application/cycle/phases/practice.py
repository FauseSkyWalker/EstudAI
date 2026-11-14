"""Practice and correction phase."""

from estudai.agents.base import AgentOutputError
from estudai.application.cycle.deps import CycleDependencies
from estudai.application.mensageria import MessageLog
from estudai.domain.correcao import Correcao, RespostaAluno, acertou
from estudai.domain.mensagens import Papel, TipoMensagem
from estudai.domain.notificacao import Notificacao, TipoNotificacao
from estudai.domain.questoes import Exercicio, OrigemItem
from estudai.ports.learner import Learner


def run_practice(
    deps: CycleDependencies,
    exercicios: list[Exercicio],
    aluno: Learner,
    log: MessageLog,
) -> tuple[list[Correcao], list[str]]:
    correcoes, removidos = [], []
    for item in exercicios:
        log.send(
            Papel.GERADOR_EXERCICIOS,
            Papel.ALUNO,
            TipoMensagem.EXERCICIO,
            item.model_dump(mode="json", include={"id", "topico", "enunciado", "alternativas"}),
            referencias=item.referencias,
        )
        resposta = aluno.solve(item.id, item)
        log.send(Papel.ALUNO, Papel.CORRETOR, TipoMensagem.RESPOSTA, resposta)
        correcao = grade_answer(deps, item, resposta)

        if correcao.gabarito_suspeito and item.origem == OrigemItem.GERADO:
            revalidacao = deps.validador.validar(item)
            log.send(Papel.PAINEL_RESOLUTORES, Papel.ORQUESTRADOR, TipoMensagem.VALIDACAO, revalidacao)
            if not revalidacao.aprovado:
                removidos.append(item.id)
                aviso = f"O exercício {item.id} foi removido: o gabarito não foi confirmado pelo painel."
                log.send(Papel.ORQUESTRADOR, Papel.ALUNO, TipoMensagem.NOTIFICACAO, {"texto": aviso})
                aluno.notify(Notificacao(tipo=TipoNotificacao.ITEM_REMOVIDO, texto=aviso, item_id=item.id))
                continue

        log.send(Papel.CORRETOR, Papel.DIAGNOSTICADOR, TipoMensagem.CORRECAO, correcao)
        aluno.notify(
            Notificacao(
                tipo=TipoNotificacao.FEEDBACK_EXERCICIO,
                texto=correcao.feedback,
                item_id=item.id,
                correta=correcao.correta,
            )
        )
        correcoes.append(correcao)
    return correcoes, removidos


def grade_answer(deps: CycleDependencies, item: Exercicio, resposta: RespostaAluno) -> Correcao:
    try:
        return deps.corretor.corrigir(item, resposta)
    except AgentOutputError:
        correta = acertou(resposta, item.gabarito)
        return Correcao(
            item_id=item.id,
            escolha=resposta.escolha,
            gabarito=item.gabarito,
            correta=correta,
            nota=float(correta),
            feedback="(análise detalhada indisponível)",
        )
