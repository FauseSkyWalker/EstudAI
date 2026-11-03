"""Aluno real interagindo pelo terminal: implementa a porta `Learner`."""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from estudai.domain.correcao import RespostaAluno
from estudai.domain.notificacao import Notificacao, TipoNotificacao
from estudai.domain.questoes import ItemObjetivo
from estudai.domain.tutoria import TurnoTutor


class ConsoleLearner:
    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def reply_to_tutor(self, turno: TurnoTutor) -> str:
        if turno.avaliacao_resposta_anterior:
            self._console.print(Panel(turno.avaliacao_resposta_anterior, title="Sobre sua resposta", style="cyan"))
        self._console.print(Panel(Markdown(turno.explicacao), title=f"Tutor · {turno.conceito_alvo}"))
        self._console.print(f"[bold]{turno.pergunta_verificacao}[/bold]")
        return Prompt.ask("Sua resposta")

    def solve(self, item_id: str, item: ItemObjetivo) -> RespostaAluno:
        # nunca exibe o gabarito
        self._console.print(Panel(item.formatar(com_gabarito=False), title=f"Exercício {item_id}"))
        escolha = Prompt.ask("Alternativa", choices=[*item.letras, *[c.lower() for c in item.letras], ""], default="")
        resolucao = Prompt.ask("Resolução (opcional, ajuda o diagnóstico)", default="")
        return RespostaAluno(item_id=item_id, escolha=escolha.upper() or None, resolucao=resolucao)

    def notify(self, notificacao: Notificacao) -> None:
        if notificacao.tipo == TipoNotificacao.FEEDBACK_EXERCICIO:
            status = "Correta" if notificacao.correta else "Incorreta"
            titulo = f"{notificacao.item_id} · {status}"
            estilo = "green" if notificacao.correta else "red"
        else:
            titulo = notificacao.tipo.value
            estilo = "yellow" if notificacao.tipo in (TipoNotificacao.ITEM_REMOVIDO, TipoNotificacao.AVISO) else "green"
        self._console.print(Panel(Markdown(notificacao.texto), title=titulo, style=estilo))
