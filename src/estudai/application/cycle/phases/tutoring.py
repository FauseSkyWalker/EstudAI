"""Tutoring phase."""

from estudai.agents.base import AgentOutputError
from estudai.agents.tools import REGISTRO_CONTEUDO, RegistroConteudo
from estudai.application.cycle.deps import CycleDependencies
from estudai.application.cycle.figures import draw_tutor_figures
from estudai.application.mensageria import MessageLog
from estudai.domain.material import PacoteMaterial
from estudai.domain.mensagens import Papel, TipoMensagem
from estudai.domain.notificacao import Notificacao, TipoNotificacao
from estudai.domain.perfil import PerfilAluno
from estudai.domain.planejamento import ObjetivoSessao
from estudai.domain.tutoria import Encerramento, TranscricaoTutoria, TrocaTutoria
from estudai.ports.learner import Learner


def run_tutoring(
    deps: CycleDependencies,
    perfil: PerfilAluno,
    objetivo: ObjetivoSessao,
    material: PacoteMaterial,
    aluno: Learner,
    log: MessageLog,
) -> tuple[TranscricaoTutoria, PacoteMaterial]:
    tutor = deps.tutores[objetivo.topico.disciplina]
    visao = perfil.visao(objetivo.topico)
    transcricao = TranscricaoTutoria(objetivo=objetivo, encerramento=Encerramento.LIMITE_RODADAS)
    alvo = set(objetivo.conceitos_alvo)
    for rodada in range(1, deps.config.max_rodadas_tutoria + 1):
        registro = RegistroConteudo(remetente=Papel.TUTOR, limite=1)
        token = REGISTRO_CONTEUDO.set(registro)
        try:
            turno = tutor.proximo_turno(visao, objetivo, material, transcricao)
        except AgentOutputError:
            material = merge_conteudo(registro, material, log)
            transcricao.encerramento = Encerramento.FALHA
            break
        finally:
            REGISTRO_CONTEUDO.reset(token)
        material = merge_conteudo(registro, material, log)
        turno = draw_tutor_figures(deps, turno)
        log.send(Papel.TUTOR, Papel.ALUNO, TipoMensagem.TUTORIA, turno, referencias=material.referencias)
        concluido = turno.objetivo_atingido or alvo <= set(transcricao.conceitos_validados + turno.conceitos_validados)
        if concluido:
            aluno.notify(Notificacao(tipo=TipoNotificacao.TUTORIA, texto=turno.explicacao, figuras=turno.imagens))
            transcricao.trocas.append(TrocaTutoria(rodada=rodada, turno=turno))
            transcricao.encerramento = Encerramento.OBJETIVO_ATINGIDO
            break
        resposta = aluno.reply_to_tutor(turno)
        log.send(Papel.ALUNO, Papel.TUTOR, TipoMensagem.RESPOSTA, {"texto": resposta})
        transcricao.trocas.append(TrocaTutoria(rodada=rodada, turno=turno, resposta_aluno=resposta))
    return transcricao, material


def merge_conteudo(registro: RegistroConteudo, material: PacoteMaterial, log: MessageLog) -> PacoteMaterial:
    for pedido, pacote in registro.trocas:
        log.send(registro.remetente, Papel.CONTEUDO, TipoMensagem.PEDIDO, {"pedido": pedido})
        log.send(
            Papel.CONTEUDO,
            registro.remetente,
            TipoMensagem.MATERIAL,
            pacote,
            referencias=pacote.referencias,
        )
        material = material.mesclar(pacote)
    return material
