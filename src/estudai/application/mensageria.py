"""Registro das mensagens do protocolo trocadas em um ciclo (auditoria e reprodutibilidade)."""

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from estudai.domain.mensagens import Mensagem, Papel, TipoMensagem


class MessageLog:
    def __init__(self) -> None:
        self._mensagens: list[Mensagem] = []

    def send(
        self,
        remetente: Papel,
        destinatario: Papel,
        tipo: TipoMensagem,
        carga: BaseModel | dict[str, Any] | None = None,
        *,
        confianca: float | None = None,
        referencias: Iterable[str] = (),
    ) -> Mensagem:
        if isinstance(carga, BaseModel):
            carga = carga.model_dump(mode="json")
        mensagem = Mensagem(
            remetente=remetente,
            destinatario=destinatario,
            tipo=tipo,
            carga=carga or {},
            confianca=confianca,
            referencias=list(referencias),
        )
        self._mensagens.append(mensagem)
        return mensagem

    @property
    def mensagens(self) -> list[Mensagem]:
        return list(self._mensagens)
