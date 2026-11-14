"""Sessões concluídas: o registro completo que permite reabrir uma sessão antiga na web.

O perfil guarda só um resumo (`EpisodioSessao`). Aqui fica o que a tela precisa para reconstruir
a conversa: as interações na ordem em que aconteceram, as figuras já renderizadas e o resultado do
ciclo. A gravação é feita só pela interface web; o núcleo não depende deste módulo.
"""

from datetime import datetime
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from estudai.application.orquestrador import ResultadoCiclo
from estudai.application.progresso import EventoProgresso
from estudai.domain.correcao import RespostaAluno
from estudai.domain.curriculum import Topico
from estudai.domain.figura import Figura
from estudai.domain.notificacao import Notificacao, TipoNotificacao
from estudai.domain.questoes import ItemObjetivo
from estudai.domain.tutoria import Encerramento, TurnoTutor


class Fase(StrEnum):
    DIAGNOSTICO = "diagnostico"
    TUTORIA = "tutoria"
    PRATICA = "pratica"


class TipoEvento(StrEnum):
    TUTOR = "tutor"  # turno do Tutor (TurnoTutor)
    ALUNO = "aluno"  # resposta do aluno ao Tutor (str)
    ITEM = "item"  # questão respondida: (item_id, ItemObjetivo, RespostaAluno)
    NOTIFICACAO = "notificacao"  # notificação tipada do sistema (Notificacao)


class ItemReplay(ItemObjetivo):
    """Item de uma sessão antiga, com as figuras já renderizadas (não só o pedido)."""

    id: str
    topico: Topico
    resolucao: str = ""
    imagens: list[Figura] = Field(default_factory=list)

    @classmethod
    def de_item(cls, item_id: str, item: ItemObjetivo) -> "ItemReplay":
        topico = getattr(item, "topico", None)
        if topico is None:
            raise ValueError("o item não tem tópico")
        return cls(
            id=item_id,
            topico=topico,
            enunciado=item.enunciado,
            alternativas=list(item.alternativas),
            gabarito=item.gabarito,
            descricoes_imagens=list(item.descricoes_imagens),
            resolucao=getattr(item, "resolucao", ""),
            imagens=list(getattr(item, "imagens", [])),
        )


class NotificacaoReplay(BaseModel):
    tipo: TipoNotificacao
    texto: str
    item_id: str | None = None
    correta: bool | None = None
    figuras: list[Figura] = Field(default_factory=list)

    @classmethod
    def de_notificacao(cls, notificacao: Notificacao) -> "NotificacaoReplay":
        return cls(
            tipo=notificacao.tipo,
            texto=notificacao.texto,
            item_id=notificacao.item_id,
            correta=notificacao.correta,
            figuras=list(notificacao.figuras),
        )

    def como_notificacao(self) -> Notificacao:
        return Notificacao(
            tipo=self.tipo,
            texto=self.texto,
            item_id=self.item_id,
            correta=self.correta,
            figuras=list(self.figuras),
        )


class EventoSessao(BaseModel):
    """Uma interação da sessão, no formato que a tela sabe desenhar."""

    fase: Fase
    tipo: TipoEvento
    turno: TurnoTutor | None = None
    turno_imagens: list[Figura] = Field(default_factory=list)
    aluno: str | None = None
    item: ItemReplay | None = None
    resposta_item: RespostaAluno | None = None
    notificacao: NotificacaoReplay | None = None
    trilha: list[EventoProgresso] = Field(default_factory=list)


class SessaoRegistrada(BaseModel):
    """Uma sessão concluída, pronta para ser reaberta em modo somente leitura."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    aluno_id: str
    topico: Topico
    iniciada_em: datetime
    finalizada_em: datetime
    resultado: ResultadoCiclo
    eventos: list[EventoSessao] = Field(default_factory=list)

    @property
    def acertos(self) -> int:
        return sum(c.correta for c in self.resultado.correcoes)

    @property
    def respondidos(self) -> int:
        return len(self.resultado.correcoes)

    @property
    def objetivo_atingido(self) -> bool:
        return self.resultado.transcricao.encerramento == Encerramento.OBJETIVO_ATINGIDO

    def resumo(self) -> "SessaoResumo":
        return SessaoResumo(
            id=self.id,
            aluno_id=self.aluno_id,
            topico=self.topico,
            iniciada_em=self.iniciada_em,
            finalizada_em=self.finalizada_em,
            acertos=self.acertos,
            respondidos=self.respondidos,
            objetivo_atingido=self.objetivo_atingido,
        )


class SessaoResumo(BaseModel):
    """Metadados leves para a lista da barra lateral."""

    id: str
    aluno_id: str
    topico: Topico
    iniciada_em: datetime
    finalizada_em: datetime
    acertos: int
    respondidos: int
    objetivo_atingido: bool


class SessionRepository(Protocol):
    def save(self, sessao: SessaoRegistrada) -> None: ...

    def load(self, aluno_id: str, sessao_id: str) -> SessaoRegistrada: ...

    def list(self, aluno_id: str) -> list[SessaoResumo]: ...
