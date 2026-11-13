"""Macrofase de tutoria: turnos do Tutor, transcrição e resumo."""

from enum import StrEnum

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from estudai.domain.curriculum import conceitos_validos
from estudai.domain.figura import Figura, PedidoFigura
from estudai.domain.planejamento import ObjetivoSessao


class TurnoTutor(BaseModel):
    """Saída estruturada de uma rodada do Agente Tutor."""

    avaliacao_resposta_anterior: str | None = None
    conceito_alvo: str
    explicacao: str
    pergunta_verificacao: str
    conceitos_validados: list[str] = Field(
        default_factory=list,
        description="Conceitos cujo entendimento o aluno já demonstrou nesta sessão.",
    )
    objetivo_atingido: bool = False
    figuras: list[PedidoFigura] = Field(
        default_factory=list,
        description=(
            "Opcional: gráficos ou diagramas que ajudem a explicação de forma visual, como, por exemplo, "
            "gráficos de funções, de cinemática, diagramas de forças e árvores de probabilidade. "
            "Escolha `tipo` grafico (matplotlib) ou diagrama (Mermaid). Lista vazia quando não for "
            "apropriado ou necessário ao ensino."
        ),
    )
    # Preenchido pelo sistema depois de desenhar `figuras` na sandbox. Fora do schema pedido ao LLM e
    # fora dos dumps (logs/mensagens), porque as imagens são pesadas.
    imagens: SkipJsonSchema[list[Figura]] = Field(default_factory=list, exclude=True)


class TrocaTutoria(BaseModel):
    rodada: int
    turno: TurnoTutor
    resposta_aluno: str | None = None


class Encerramento(StrEnum):
    OBJETIVO_ATINGIDO = "objetivo_atingido"
    LIMITE_RODADAS = "limite_rodadas"
    FALHA = "falha"


class TranscricaoTutoria(BaseModel):
    objetivo: ObjetivoSessao
    trocas: list[TrocaTutoria] = Field(default_factory=list)
    encerramento: Encerramento | None = None

    @property
    def conceitos_validados(self) -> list[str]:
        validados = [c for t in self.trocas for c in t.turno.conceitos_validados]
        return conceitos_validos(self.objetivo.topico, validados)

    def como_texto(self) -> str:
        linhas = []
        for t in self.trocas:
            if t.turno.avaliacao_resposta_anterior:
                linhas.append(f"TUTOR (avaliação): {t.turno.avaliacao_resposta_anterior}")
            linhas.append(f"TUTOR [{t.turno.conceito_alvo}]: {t.turno.explicacao}")
            for figura in t.turno.figuras:  # juízes e resumo precisam saber quais figuras foram mostradas
                linhas.append(f"TUTOR (figura): {figura.descricao}")
            linhas.append(f"TUTOR (pergunta): {t.turno.pergunta_verificacao}")
            if t.resposta_aluno is not None:
                linhas.append(f"ALUNO: {t.resposta_aluno}")
        return "\n".join(linhas)


class ResumoEstudo(BaseModel):
    """Saída estruturada do Agente de Resumo após a tutoria (material de revisão)."""

    titulo: str
    pontos_chave: list[str] = Field(min_length=1)
    formulas: list[str] = Field(default_factory=list)
    erros_comuns: list[str] = Field(default_factory=list)
    proximos_passos: list[str] = Field(default_factory=list)
