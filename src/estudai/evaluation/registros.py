"""Condições e registros (schemas) do protocolo experimental."""

from enum import StrEnum

from pydantic import BaseModel, Field

from estudai.application.pratica import ItemDescartado
from estudai.application.telemetria import UsageSummary, UsageTotals
from estudai.application.validacao import ResultadoValidacao
from estudai.domain.curriculum import Topico
from estudai.domain.perfil import NivelDominio
from estudai.evaluation.juizes import AvaliacaoItem, NotasTutoria, VeredictoTransicao


class Condicao(StrEnum):
    C1 = "C1"  # sistema completo e adaptativo
    C2 = "C2"  # perfil congelado (isola o valor da adaptatividade)
    C3 = "C3"  # somente questões reais do banco (referência para a geração)

    @property
    def adaptativo(self) -> bool:
        return self != Condicao.C2

    @property
    def fonte(self) -> str:
        return "banco" if self == Condicao.C3 else "gerado"


class ItemRegistro(BaseModel):
    id: str
    origem: str
    conceitos: list[str]
    dificuldade: str
    enunciado: str
    alternativas: list[str]
    gabarito: str
    descricoes_imagens: list[str] = Field(default_factory=list)
    resolucao: str = ""
    codigo_verificacao: str | None = None
    referencias: list[str] = Field(default_factory=list)
    avaliacoes: list[AvaliacaoItem] = Field(default_factory=list)
    votos_painel: list[str] = Field(default_factory=list)


class ItemDescartadoRegistro(BaseModel):
    id: str | None = None
    origem: str | None = None
    conceitos: list[str] = Field(default_factory=list)
    dificuldade: str | None = None
    enunciado: str | None = None
    alternativas: list[str] = Field(default_factory=list)
    gabarito: str | None = None
    descricoes_imagens: list[str] = Field(default_factory=list)
    resolucao: str = ""
    codigo_verificacao: str | None = None
    referencias: list[str] = Field(default_factory=list)
    motivo: str
    validacao: ResultadoValidacao | None = None


class TransicaoRegistro(BaseModel):
    conceito: str
    na_lacuna: bool
    aprendido: bool
    erro: str | None = None
    avaliacoes: list[VeredictoTransicao | None] = Field(default_factory=list)


class RegistroCiclo(BaseModel):
    ciclo: int
    nivel_verdadeiro_inicial: NivelDominio
    nivel_verdadeiro_final: NivelDominio
    nivel_estimado_inicial: NivelDominio | None
    nivel_estimado_final: NivelDominio | None
    encerramento_tutoria: str | None
    conceitos_aprendidos: list[str]
    transicoes: list[TransicaoRegistro] = Field(default_factory=list)
    notas_tutoria: list[NotasTutoria]  # uma por juiz
    avaliacoes_itens: list[list[AvaliacaoItem]]  # [item][juiz]
    votos_painel: list[list[str]]  # [item aprovado][resolutor]
    itens: list[ItemRegistro] = Field(default_factory=list)
    descartados: list[ItemDescartadoRegistro] = Field(default_factory=list)
    uso_por_agente: dict[str, UsageTotals] = Field(default_factory=dict)
    itens_entregues: int
    itens_removidos: int
    itens_descartados: int
    taxa_descarte: float
    acertos_pratica: int
    custo_usd: float
    tokens_entrada: int
    tokens_saida: int
    latencia_s: float
    mensagens: int = 0
    arquivo_mensagens: str | None = None
    arquivo_usage: str | None = None


class ResultadoExecucao(BaseModel):
    condicao: Condicao
    persona_id: str
    topico: Topico
    seed: int
    pre_teste: float
    pos_teste: float
    ciclos: list[RegistroCiclo] = Field(default_factory=list)
    uso_avaliacao: UsageSummary = Field(default_factory=UsageSummary)
    erro: str | None = None

    @property
    def chave(self) -> str:
        return f"{self.condicao.value}|{self.persona_id}|{self.topico.value}|{self.seed}"

    @property
    def ganho(self) -> float:
        return self.pos_teste - self.pre_teste


def registro_descartado(item: ItemDescartado) -> ItemDescartadoRegistro:
    """Snapshot do item que o filtro descartou, para auditar o que a geração produziu."""
    exercicio = item.exercicio
    if exercicio is None:
        return ItemDescartadoRegistro(motivo=item.motivo)
    return ItemDescartadoRegistro(
        id=exercicio.id,
        origem=exercicio.origem.value,
        conceitos=exercicio.conceitos,
        dificuldade=exercicio.dificuldade_alvo.value,
        enunciado=exercicio.enunciado,
        alternativas=exercicio.alternativas,
        gabarito=exercicio.gabarito,
        descricoes_imagens=exercicio.descricoes_imagens,
        resolucao=exercicio.resolucao,
        codigo_verificacao=exercicio.codigo_verificacao,
        referencias=exercicio.referencias,
        motivo=item.motivo,
        validacao=item.validacao,
    )
