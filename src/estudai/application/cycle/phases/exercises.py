"""Exercise generation phase."""

from estudai.agents.tools import REGISTRO_CONTEUDO, RegistroConteudo
from estudai.application.cycle.deps import CycleDependencies
from estudai.application.cycle.figures import draw_exercise_figures
from estudai.application.mensageria import MessageLog
from estudai.application.pratica import ResultadoFornecimento
from estudai.application.progresso import EVENTO_NOTA, notificar_progresso
from estudai.domain.curriculum import Topico
from estudai.domain.mensagens import Papel, TipoMensagem
from estudai.domain.perfil import PerfilAluno


def provide_exercises(
    deps: CycleDependencies,
    perfil: PerfilAluno,
    topico: Topico,
    excluir: frozenset[str],
    log: MessageLog,
) -> ResultadoFornecimento:
    registro = RegistroConteudo(remetente=Papel.GERADOR_EXERCICIOS)
    token = REGISTRO_CONTEUDO.set(registro)
    try:
        fornecimento = deps.fornecedor.fornecer(perfil, topico, deps.config.exercicios_por_sessao, excluir)
    finally:
        REGISTRO_CONTEUDO.reset(token)
    for pedido, pacote in registro.trocas:
        log.send(Papel.GERADOR_EXERCICIOS, Papel.CONTEUDO, TipoMensagem.PEDIDO, {"pedido": pedido})
        log.send(
            Papel.CONTEUDO,
            Papel.GERADOR_EXERCICIOS,
            TipoMensagem.MATERIAL,
            pacote,
            referencias=pacote.referencias,
        )
    for descartado in fornecimento.descartados:
        log.send(
            Papel.PAINEL_RESOLUTORES,
            Papel.GERADOR_EXERCICIOS,
            TipoMensagem.VALIDACAO,
            {"aprovado": False, "motivo": descartado.motivo},
        )
    total = len(fornecimento.exercicios)
    com_figura = sum(bool(exercicio.figuras) for exercicio in fornecimento.exercicios)
    notificar_progresso(
        {
            "agente": "Gerador de Exercícios",
            "descricao": f"{com_figura}/{total} questões geradas pediram figura",
        },
        evento=EVENTO_NOTA,
    )
    exercicios = [draw_exercise_figures(deps, exercicio) for exercicio in fornecimento.exercicios]
    return fornecimento.model_copy(update={"exercicios": exercicios})
