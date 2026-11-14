"""Initial material retrieval."""

from estudai.application.cycle.deps import CycleDependencies
from estudai.application.mensageria import MessageLog
from estudai.domain.material import PacoteMaterial
from estudai.domain.mensagens import Papel, TipoMensagem
from estudai.domain.perfil import PerfilAluno
from estudai.domain.planejamento import ObjetivoSessao


def retrieve_material(
    deps: CycleDependencies,
    perfil: PerfilAluno,
    objetivo: ObjetivoSessao,
    log: MessageLog,
) -> PacoteMaterial:
    nivel = objetivo.nivel_aluno.value if objetivo.nivel_aluno else "não diagnosticado"
    pedido = f"{objetivo.descricao} Nível do aluno: {nivel}."
    log.send(Papel.TUTOR, Papel.CONTEUDO, TipoMensagem.PEDIDO, {"pedido": pedido})
    material = deps.conteudo.buscar(pedido, objetivo.topico)
    log.send(Papel.CONTEUDO, Papel.TUTOR, TipoMensagem.MATERIAL, material, referencias=material.referencias)
    return material
