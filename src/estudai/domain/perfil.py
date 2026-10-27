"""Perfil persistente do aluno: estado único, centralizado e versionado (J. David & S. Ghosh, 2026).

O perfil é imutável: toda atualização produz uma nova versão, o que permite auditar a
evolução do diagnóstico e reproduzir o estado visto por cada agente em cada etapa.
"""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from estudai.domain.curriculum import Topico, conceitos_validos


class NivelDominio(StrEnum):
    INICIANTE = "iniciante"
    INTERMEDIARIO = "intermediario"
    AVANCADO = "avancado"

    @classmethod
    def da_fracao(cls, fracao_dominada: float) -> "NivelDominio":
        if fracao_dominada < 1 / 3:
            return cls.INICIANTE
        if fracao_dominada < 2 / 3:
            return cls.INTERMEDIARIO
        return cls.AVANCADO


class TipoErro(StrEnum):
    CONCEITUAL = "conceitual"
    NUMERICO = "numerico"
    INTERPRETACAO = "interpretacao"


class RegistroErro(BaseModel):
    model_config = ConfigDict(frozen=True)

    topico: Topico
    item_id: str
    tipo: TipoErro
    conceito: str | None = None
    descricao: str


class Diagnostico(BaseModel):
    """Saída estruturada do Agente Diagnosticador para um tópico."""

    topico: Topico
    nivel: NivelDominio
    conceitos_dominados: list[str] = Field(default_factory=list)
    lacunas: list[str] = Field(default_factory=list)
    justificativa: str = ""
    confianca: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _normalizar_conceitos(self):
        self.lacunas = conceitos_validos(self.topico, self.lacunas)
        lacunas = set(self.lacunas)
        self.conceitos_dominados = [
            c for c in conceitos_validos(self.topico, self.conceitos_dominados) if c not in lacunas
        ]
        return self


class EstadoTopico(BaseModel):
    model_config = ConfigDict(frozen=True)

    nivel: NivelDominio | None = None  # None = ainda não diagnosticado
    confianca: float = 0.0
    conceitos_dominados: tuple[str, ...] = ()
    lacunas: tuple[str, ...] = ()
    itens_respondidos: int = 0
    acertos: int = 0

    @property
    def taxa_acerto(self) -> float | None:
        return self.acertos / self.itens_respondidos if self.itens_respondidos else None


class PerfilAluno(BaseModel):
    model_config = ConfigDict(frozen=True)

    aluno_id: str
    versao: int = 0
    topicos: dict[Topico, EstadoTopico] = Field(default_factory=dict)
    historico_erros: tuple[RegistroErro, ...] = ()
    motivo: str = "criacao"
    atualizado_em: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def novo(cls, aluno_id: str) -> "PerfilAluno":
        return cls(aluno_id=aluno_id)

    def estado(self, topico: Topico) -> EstadoTopico:
        return self.topicos.get(topico, EstadoTopico())

    def diagnosticado(self, topico: Topico) -> bool:
        return self.estado(topico).nivel is not None

    def evoluir(
        self,
        motivo: str,
        diagnostico: Diagnostico | None = None,
        novos_erros: tuple[RegistroErro, ...] = (),
        acertos: int = 0,
        respondidos: int = 0,
        topico: Topico | None = None,
    ) -> "PerfilAluno":
        """Retorna a próxima versão do perfil com o diagnóstico e as evidências aplicados."""
        topico = diagnostico.topico if diagnostico else topico
        topicos = dict(self.topicos)
        if topico is not None:
            atual = self.estado(topico)
            atualizacao: dict = {
                "itens_respondidos": atual.itens_respondidos + respondidos,
                "acertos": atual.acertos + acertos,
            }
            if diagnostico:
                atualizacao |= {
                    "nivel": diagnostico.nivel,
                    "confianca": diagnostico.confianca,
                    "conceitos_dominados": tuple(diagnostico.conceitos_dominados),
                    "lacunas": tuple(diagnostico.lacunas),
                }
            topicos[topico] = atual.model_copy(update=atualizacao)
        return self.model_copy(
            update={
                "versao": self.versao + 1,
                "topicos": topicos,
                "historico_erros": self.historico_erros + tuple(novos_erros),
                "motivo": motivo,
                "atualizado_em": datetime.now(UTC),
            }
        )

    def visao(self, topico: Topico, max_erros: int = 8) -> dict:
        """Recorte compacto do perfil para compor prompts (economiza janela de contexto)."""
        estado = self.estado(topico)
        erros = [e for e in self.historico_erros if e.topico == topico][-max_erros:]
        return {
            "aluno_id": self.aluno_id,
            "versao_perfil": self.versao,
            "topico": topico.value,
            "nivel": estado.nivel.value if estado.nivel else "nao_diagnosticado",
            "confianca": round(estado.confianca, 2),
            "conceitos_dominados": list(estado.conceitos_dominados),
            "lacunas": list(estado.lacunas),
            "taxa_acerto": estado.taxa_acerto,
            "erros_recentes": [e.model_dump(mode="json", exclude={"topico"}) for e in erros],
        }
