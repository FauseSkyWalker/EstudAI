"""Protocolo de comunicação entre agentes: mensagens JSON tipadas (Seção 4 e Apêndice A)."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Papel(StrEnum):
    ORQUESTRADOR = "orquestrador"
    DIAGNOSTICADOR = "diagnosticador"
    TUTOR = "tutor"
    CONTEUDO = "conteudo"
    RESUMO = "resumo"
    GERADOR_EXERCICIOS = "gerador_exercicios"
    CORRETOR = "corretor"
    PAINEL_RESOLUTORES = "painel_resolutores"
    ALUNO = "aluno"
    # papéis fora do pipeline (pré-processamento e avaliação), usados na telemetria de custo
    CLASSIFICADOR = "classificador"
    ALUNO_SIMULADO = "aluno_simulado"
    JUIZ = "juiz"


class TipoMensagem(StrEnum):
    PEDIDO = "pedido"
    DIAGNOSTICO = "diagnostico"
    MATERIAL = "material"
    TUTORIA = "tutoria"
    RESUMO = "resumo"
    EXERCICIO = "exercicio"
    VALIDACAO = "validacao"
    RESPOSTA = "resposta"
    CORRECAO = "correcao"
    PERFIL = "perfil"
    NOTIFICACAO = "notificacao"


class Mensagem(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    remetente: Papel
    destinatario: Papel
    tipo: TipoMensagem
    carga: dict[str, Any] = Field(default_factory=dict)
    confianca: float | None = Field(default=None, ge=0.0, le=1.0)
    referencias: list[str] = Field(default_factory=list)
    criada_em: datetime = Field(default_factory=lambda: datetime.now(UTC))
