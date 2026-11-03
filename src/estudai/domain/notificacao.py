from enum import StrEnum

from pydantic import BaseModel


class TipoNotificacao(StrEnum):
    TUTORIA = "tutoria"
    RESUMO = "resumo"
    FEEDBACK_EXERCICIO = "feedback_exercicio"
    ITEM_REMOVIDO = "item_removido"
    RELATORIO = "relatorio"
    AVISO = "aviso"


class Notificacao(BaseModel):
    tipo: TipoNotificacao
    texto: str
    item_id: str | None = None
    correta: bool | None = None
