"""Figure rendering for tutoring turns and exercises."""

from estudai.application.cycle.deps import CycleDependencies
from estudai.application.progresso import EVENTO_FIGURA, notificar_progresso
from estudai.domain.figura import Figura, PedidoFigura
from estudai.domain.questoes import Exercicio
from estudai.domain.tutoria import TurnoTutor


def draw_tutor_figures(deps: CycleDependencies, turno: TurnoTutor) -> TurnoTutor:
    if not turno.figuras:
        return turno
    if deps.figuras is None:
        return turno.model_copy(update={"figuras": []})

    aceitos: list[PedidoFigura] = []
    imagens: list[Figura] = []
    for indice, pedido in enumerate(turno.figuras[: max(0, deps.max_figuras)], start=1):
        notificar_progresso(
            {"agente": "Tutor", "descricao": f"desenhando figura {indice}: {pedido.descricao}"},
            evento=EVENTO_FIGURA,
        )
        try:
            imagem = deps.figuras.render(pedido)
        except Exception as exc:  # noqa: BLE001 - figura nunca deve descartar o ciclo
            notificar_progresso(
                {"agente": "Tutor", "descricao": f"figura {indice} descartada ({str(exc)[:120]})"},
                evento=EVENTO_FIGURA,
            )
            continue
        aceitos.append(pedido)
        imagens.append(imagem)
    return turno.model_copy(update={"figuras": aceitos, "imagens": imagens})


def draw_exercise_figures(deps: CycleDependencies, exercicio: Exercicio) -> Exercicio:
    if not exercicio.figuras:
        return exercicio
    pedidos = exercicio.figuras[: max(0, deps.max_figuras_exercicio)]
    descricoes = [pedido.descricao for pedido in pedidos]
    if deps.figuras is None:
        return exercicio.model_copy(update={"figuras": [], "imagens": [], "descricoes_imagens": []})

    imagens: list[Figura] = []
    for indice, pedido in enumerate(pedidos, start=1):
        notificar_progresso(
            {
                "agente": "Gerador de Exercícios",
                "descricao": f"desenhando figura {indice} da questão {exercicio.id}: {pedido.descricao}",
            },
            evento=EVENTO_FIGURA,
        )
        try:
            imagens.append(deps.figuras.render(pedido))
        except Exception as exc:  # noqa: BLE001 - figura não pode descartar a questão
            notificar_progresso(
                {
                    "agente": "Gerador de Exercícios",
                    "descricao": f"figura {indice} da questão {exercicio.id} descartada ({str(exc)[:120]})",
                },
                evento=EVENTO_FIGURA,
            )
    return exercicio.model_copy(update={"figuras": pedidos, "imagens": imagens, "descricoes_imagens": descricoes})
