"""Aluno real interagindo pelo terminal: implementa a porta `Learner`."""

from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from estudai.application.progresso import EventoProgresso, TipoProgresso
from estudai.domain.correcao import RespostaAluno
from estudai.domain.curriculum import rotulo_conceito
from estudai.domain.figura import Figura, FormatoFigura
from estudai.domain.notificacao import Notificacao, TipoNotificacao
from estudai.domain.questoes import ItemObjetivo
from estudai.domain.tutoria import TurnoTutor


class ConsoleLearner:
    def __init__(self, console: Console | None = None, pasta_figuras: Path | None = None) -> None:
        """`pasta_figuras`: onde salvar as figuras do Tutor (o terminal não exibe imagens)."""
        self._console = console or Console()
        self._pasta_figuras = pasta_figuras
        self._figuras_salvas = 0

    def bastidores(self, evento: EventoProgresso) -> None:
        """Observador de progresso: uma linha discreta por ação dos agentes."""
        if evento.tipo == TipoProgresso.ETAPA:
            self._console.print(f"[dim]── {evento.descricao}[/dim]")
        elif evento.tipo != TipoProgresso.RESPONDEU:
            quem = f"{evento.agente} " if evento.agente else ""
            self._console.print(f"[dim]   {quem}{evento.descricao}[/dim]")

    def _mostrar_figura(self, figura: Figura | None) -> None:
        if figura is None:
            return
        if figura.formato != FormatoFigura.PNG:
            return  # Mermaid apenas na interface web
        if self._pasta_figuras is None:
            self._console.print(f"[magenta]🖼️  Figura: {figura.descricao}[/magenta]")
            return
        self._pasta_figuras.mkdir(parents=True, exist_ok=True)
        self._figuras_salvas += 1
        arquivo = self._pasta_figuras / f"figura-{datetime.now():%Y%m%d-%H%M%S}-{self._figuras_salvas}.png"
        arquivo.write_bytes(figura.png)
        self._console.print(f"[magenta]🖼️  {figura.descricao} → abra a imagem: {arquivo.resolve()}[/magenta]")

    def reply_to_tutor(self, turno: TurnoTutor) -> str:
        if turno.avaliacao_resposta_anterior:
            self._console.print(Panel(turno.avaliacao_resposta_anterior, title="Sobre sua resposta", style="cyan"))
        self._console.print(Panel(Markdown(turno.explicacao), title=f"Tutor · {rotulo_conceito(turno.conceito_alvo)}"))
        for figura in turno.imagens:
            self._mostrar_figura(figura)
        self._console.print(f"[bold]{turno.pergunta_verificacao}[/bold]")
        return Prompt.ask("Sua resposta")

    def solve(self, item_id: str, item: ItemObjetivo) -> RespostaAluno:
        # nunca exibe o gabarito
        self._console.print(Panel(item.formatar(com_gabarito=False), title=f"Exercício {item_id}"))
        for figura in getattr(item, "imagens", []):
            self._mostrar_figura(figura)
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
        for figura in notificacao.figuras:
            self._mostrar_figura(figura)
