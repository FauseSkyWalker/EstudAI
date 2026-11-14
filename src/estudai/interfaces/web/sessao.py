"""Ponte entre o orquestrador e uma interface web.

O orquestrador é síncrono: dentro de `executar_ciclo`, ele bloqueia esperando a resposta do aluno.
Uma interface web, ao contrário, reexecuta a cada clique. A ponte roda o ciclo numa thread e expõe
um `Learner` (`WebLearner`) que, quando o orquestrador pede algo ao aluno, publica o pedido e espera
a resposta numa fila. A tela só lê o estado e entrega respostas; o núcleo não muda nada.
"""

import queue
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from estudai.application.orquestrador import Orquestrador, ResultadoCiclo
from estudai.application.progresso import EventoProgresso
from estudai.domain.correcao import RespostaAluno
from estudai.domain.curriculum import Topico
from estudai.domain.notificacao import Notificacao
from estudai.domain.questoes import ItemObjetivo, Questao
from estudai.domain.tutoria import TurnoTutor


class Fase(StrEnum):
    DIAGNOSTICO = "diagnostico"
    TUTORIA = "tutoria"
    PRATICA = "pratica"


class TipoEvento(StrEnum):
    TUTOR = "tutor"  # turno do Tutor (TurnoTutor)
    ALUNO = "aluno"  # resposta do aluno ao Tutor (str)
    ITEM = "item"  # questão respondida: (item_id, ItemObjetivo, RespostaAluno)
    NOTIFICACAO = "notificacao"  # notificação tipada do sistema (Notificacao)


@dataclass(frozen=True)
class Evento:
    tipo: TipoEvento
    fase: Fase
    conteudo: Any
    # Bastidores que produziram esta mensagem (progresso desde a mensagem anterior)
    trilha: tuple[EventoProgresso, ...] = ()


@dataclass(frozen=True)
class Pedido:
    """O que o orquestrador está esperando do aluno neste momento."""

    fase: Fase
    turno: TurnoTutor | None = None  # pedido de resposta ao Tutor
    item_id: str | None = None  # pedido de resolução de questão
    item: ItemObjetivo | None = None
    trilha: tuple[EventoProgresso, ...] = ()  # bastidores que produziram o pedido


class SessaoCancelada(Exception):
    """Levantada dentro da thread do orquestrador quando o aluno encerra a sessão."""


_CANCELAR = object()


class WebLearner:
    """Implementa a porta `Learner`; thread-safe entre a thread do ciclo e a da interface."""

    def __init__(self, progresso: Callable[[], list[EventoProgresso]] | None = None) -> None:
        self._lock = threading.Lock()
        self._eventos: list[Evento] = []
        self._pendente: Pedido | None = None
        self._respostas: queue.Queue = queue.Queue()
        self._fase = Fase.DIAGNOSTICO
        self._progresso = progresso or (lambda: [])
        self._progresso_consumido = 0

    # --- lado do orquestrador (thread do ciclo) ------------------------------------------

    def reply_to_tutor(self, turno: TurnoTutor) -> str:
        self._fase = Fase.TUTORIA
        self._registrar(TipoEvento.TUTOR, turno)
        resposta, _ = self._aguardar(Pedido(fase=Fase.TUTORIA, turno=turno))
        self._registrar(TipoEvento.ALUNO, resposta)
        return resposta

    def solve(self, item_id: str, item: ItemObjetivo) -> RespostaAluno:
        # o diagnóstico usa questões do banco; a prática entrega `Exercicio`
        self._fase = Fase.DIAGNOSTICO if isinstance(item, Questao) else Fase.PRATICA
        resposta, pedido = self._aguardar(Pedido(fase=self._fase, item_id=item_id, item=item))
        self._registrar(TipoEvento.ITEM, (item_id, item, resposta), trilha=pedido.trilha)
        return resposta

    def notify(self, notificacao: Notificacao) -> None:
        self._registrar(TipoEvento.NOTIFICACAO, notificacao)

    # --- lado da interface ---------------------------------------------------------------

    @property
    def pendente(self) -> Pedido | None:
        with self._lock:
            return self._pendente

    def eventos(self) -> list[Evento]:
        with self._lock:
            return list(self._eventos)

    @property
    def progresso_consumido(self) -> int:
        with self._lock:
            return self._progresso_consumido

    def responder(self, resposta: str | RespostaAluno) -> None:
        with self._lock:
            if self._pendente is None:
                raise RuntimeError("nenhum pedido pendente")
            self._pendente = None
        self._respostas.put(resposta)

    def cancelar(self) -> None:
        with self._lock:
            self._pendente = None
        self._respostas.put(_CANCELAR)

    # --- interno -------------------------------------------------------------------------

    def _registrar(
        self,
        tipo: TipoEvento,
        conteudo: Any,
        *,
        trilha: tuple[EventoProgresso, ...] | None = None,
    ) -> None:
        with self._lock:
            if trilha is None:
                trilha = self._consumir_trilha()
            self._eventos.append(Evento(tipo=tipo, fase=self._fase, conteudo=conteudo, trilha=trilha))

    def _aguardar(self, pedido: Pedido) -> tuple[Any, Pedido]:
        with self._lock:
            pedido = replace(pedido, trilha=self._consumir_trilha())
            self._pendente = pedido
        resposta = self._respostas.get()
        if resposta is _CANCELAR:
            raise SessaoCancelada
        return resposta, pedido

    def _consumir_trilha(self) -> tuple[EventoProgresso, ...]:
        """Progresso ainda não anexado a uma mensagem (o lock do learner deve estar tomado)."""
        eventos = self._progresso()
        novos = tuple(eventos[self._progresso_consumido :])
        self._progresso_consumido = len(eventos)
        return novos


class EstadoSessao(StrEnum):
    TRABALHANDO = "trabalhando"  # agentes processando (chamadas de LLM)
    AGUARDANDO_ALUNO = "aguardando_aluno"
    CONCLUIDA = "concluida"
    CANCELADA = "cancelada"
    ERRO = "erro"


class SessaoWeb:
    """Um ciclo de estudo rodando em segundo plano."""

    def __init__(self, orquestrador: Orquestrador, aluno_id: str, topico: Topico) -> None:
        self.aluno_id = aluno_id
        self.topico = topico
        self._progresso: list[EventoProgresso] = []  # bastidores: o que cada agente está fazendo
        self._lock_progresso = threading.Lock()
        self.learner = WebLearner(self.progresso)
        self.resultado: ResultadoCiclo | None = None
        self.erro: str | None = None
        self._cancelada = False
        self._orquestrador = orquestrador
        self._thread = threading.Thread(target=self._rodar, name=f"ciclo-{aluno_id}", daemon=True)

    def iniciar(self) -> "SessaoWeb":
        self._thread.start()
        return self

    def cancelar(self) -> None:
        self._cancelada = True
        self.learner.cancelar()

    def aguardar(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def progresso(self) -> list[EventoProgresso]:
        with self._lock_progresso:
            return list(self._progresso)

    def progresso_pendente(self) -> list[EventoProgresso]:
        """O progresso ainda não anexado a uma mensagem (o que está sendo gerado agora)."""
        consumido = self.learner.progresso_consumido
        with self._lock_progresso:
            return list(self._progresso[consumido:])

    def _registrar_progresso(self, evento: EventoProgresso) -> None:
        # chamado pela thread do ciclo e pelas threads do painel de resolutores
        with self._lock_progresso:
            self._progresso.append(evento)

    @property
    def estado(self) -> EstadoSessao:
        if self._thread.is_alive():
            return EstadoSessao.AGUARDANDO_ALUNO if self.learner.pendente else EstadoSessao.TRABALHANDO
        if self._cancelada:
            return EstadoSessao.CANCELADA
        return EstadoSessao.ERRO if self.erro else EstadoSessao.CONCLUIDA

    def _rodar(self) -> None:
        try:
            self.resultado = self._orquestrador.executar_ciclo(
                self.aluno_id, self.topico, self.learner, observador=self._registrar_progresso
            )
        except SessaoCancelada:
            self._cancelada = True
        except Exception as exc:  # noqa: BLE001 - o erro é mostrado na tela, não pode morrer calado
            self.erro = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=5)}"
