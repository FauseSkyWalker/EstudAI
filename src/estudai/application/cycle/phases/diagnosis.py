"""Initial diagnosis phase."""

from estudai.agents.diagnosticador import Evidencia
from estudai.application.cycle.deps import CycleDependencies
from estudai.application.mensageria import MessageLog
from estudai.domain.correcao import acertou
from estudai.domain.curriculum import Topico
from estudai.domain.mensagens import Papel, TipoMensagem
from estudai.domain.perfil import Diagnostico, PerfilAluno
from estudai.domain.questoes import Particao
from estudai.ports.learner import Learner


def run_initial_diagnosis(
    deps: CycleDependencies,
    perfil: PerfilAluno,
    topico: Topico,
    aluno: Learner,
    log: MessageLog,
) -> tuple[PerfilAluno, Diagnostico | None, list[str]]:
    banco = deps.banco.find(topico, Particao.BANCO)
    sondagem = deps.diagnosticador.selecionar_sondagem(topico, banco, seed=deps.config.seed)
    evidencias = []
    for questao in sondagem:
        log.send(Papel.DIAGNOSTICADOR, Papel.ALUNO, TipoMensagem.PEDIDO, {"item_id": questao.id})
        resposta = aluno.solve(questao.id, questao)
        log.send(Papel.ALUNO, Papel.DIAGNOSTICADOR, TipoMensagem.RESPOSTA, resposta)
        evidencias.append(
            Evidencia(
                item_id=questao.id,
                conceitos=questao.conceitos,
                dificuldade=questao.dificuldade,
                correta=acertou(resposta, questao.gabarito),
            )
        )
    if not evidencias:
        return perfil, None, []
    diagnostico = deps.diagnosticador.estimar(perfil, topico, evidencias)
    perfil = perfil.evoluir(
        "diagnostico_inicial",
        diagnostico=diagnostico,
        acertos=sum(e.correta for e in evidencias),
        respondidos=len(evidencias),
    )
    deps.perfis.save(perfil)
    log.send(
        Papel.DIAGNOSTICADOR,
        Papel.ORQUESTRADOR,
        TipoMensagem.DIAGNOSTICO,
        diagnostico,
        confianca=diagnostico.confianca,
        referencias=[q.id for q in sondagem],
    )
    return perfil, diagnostico, [q.id for q in sondagem]
