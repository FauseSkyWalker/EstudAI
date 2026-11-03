from typing import Protocol

from estudai.domain.correcao import RespostaAluno
from estudai.domain.notificacao import Notificacao
from estudai.domain.questoes import ItemObjetivo
from estudai.domain.tutoria import TurnoTutor


class Learner(Protocol):
    """Quem estuda: um usuário real (CLI) ou um estudante simulado (avaliação).

    O orquestrador conversa apenas com esta interface, então o mesmo pipeline roda
    em produção e nos experimentos sem nenhuma ramificação.
    """

    def reply_to_tutor(self, turno: TurnoTutor) -> str: ...

    def solve(self, item_id: str, item: ItemObjetivo) -> RespostaAluno: ...

    def notify(self, notificacao: Notificacao) -> None: ...
