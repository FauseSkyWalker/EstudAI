"""Itens objetivos: questões do banco (gabarito oficial) e exercícios entregues ao aluno."""

import random
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from estudai.domain.curriculum import Disciplina, Topico, conceitos_validos
from estudai.domain.figura import Figura, PedidoFigura

LETRAS = ("A", "B", "C", "D", "E")


class Dificuldade(StrEnum):
    FACIL = "facil"
    MEDIA = "media"
    DIFICIL = "dificil"


class Particao(StrEnum):
    """`bateria` = régua fixa de pré/pós-teste; nunca é exposta aos agentes (evita vazamento)."""

    BANCO = "banco"
    BATERIA = "bateria"


class OrigemItem(StrEnum):
    GERADO = "gerado"
    BANCO = "banco"


class ItemObjetivo(BaseModel):
    """Base comum: enunciado, 4 ou 5 alternativas e gabarito por letra."""

    enunciado: str
    alternativas: list[str] = Field(min_length=4, max_length=5)
    gabarito: str
    descricoes_imagens: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _gabarito_valido(self):
        self.gabarito = self.gabarito.strip().upper()
        if self.gabarito not in self.letras:
            raise ValueError(f"gabarito '{self.gabarito}' fora de {self.letras}")
        return self

    @property
    def letras(self) -> tuple[str, ...]:
        return LETRAS[: len(self.alternativas)]

    def alternativa(self, letra: str) -> str:
        return self.alternativas[self.letras.index(letra.upper())]

    def formatar(self, com_gabarito: bool = False) -> str:
        linhas = [self.enunciado]
        for descricao in self.descricoes_imagens:
            marcador = f"[Figura: {descricao}]"
            if marcador.lower() not in self.enunciado.lower():
                linhas.append(marcador)
        linhas.append("")
        linhas += [f"{letra}) {texto}" for letra, texto in zip(self.letras, self.alternativas, strict=True)]
        if com_gabarito:
            linhas += ["", f"Gabarito: {self.gabarito}"]
        return "\n".join(linhas)


class Questao(ItemObjetivo):
    """Questão real de vestibular, normalizada a partir das bases públicas (Tabela 1)."""

    model_config = ConfigDict(frozen=False)

    id: str
    fonte: str
    ano: int | None = None
    disciplina: Disciplina
    topico: Topico
    conceitos: list[str] = Field(default_factory=list)
    dificuldade: Dificuldade = Dificuldade.MEDIA
    particao: Particao = Particao.BANCO

    @model_validator(mode="after")
    def _conceitos_do_topico(self):
        self.conceitos = conceitos_validos(self.topico, self.conceitos)
        return self


class Exercicio(ItemObjetivo):
    """Item entregue ao aluno na macrofase de prática: gerado pelo sistema ou vindo do banco."""

    id: str
    topico: Topico
    conceitos: list[str] = Field(default_factory=list)
    dificuldade_alvo: Dificuldade = Dificuldade.MEDIA
    resolucao: str = ""
    origem: OrigemItem = OrigemItem.GERADO
    referencias: list[str] = Field(default_factory=list)
    codigo_verificacao: str | None = None
    figuras: list[PedidoFigura] = Field(default_factory=list)
    # Preenchido depois de renderizar `figuras`; fora do schema e dos dumps (imagens são pesadas).
    imagens: SkipJsonSchema[list[Figura]] = Field(default_factory=list, exclude=True)

    @model_validator(mode="after")
    def _conceitos_do_topico(self):
        self.conceitos = conceitos_validos(self.topico, self.conceitos)
        return self

    def com_alternativas_embaralhadas(self) -> "Exercicio":
        """Reordena as alternativas e remapeia o gabarito.

        LLMs concentram a resposta certa em certas posições (no teste real, os 4 itens gerados
        tinham gabarito C), o que um aluno aprenderia a explorar. O sorteio deriva do id do
        exercício, então a mesma questão sempre sai na mesma ordem (reprodutível).
        """
        ordem = list(range(len(self.alternativas)))
        random.Random(self.id).shuffle(ordem)
        correta = self.letras.index(self.gabarito)
        return self.model_copy(
            update={
                "alternativas": [self.alternativas[i] for i in ordem],
                "gabarito": LETRAS[ordem.index(correta)],
            }
        )

    @classmethod
    def da_questao(cls, questao: Questao) -> "Exercicio":
        return cls(
            id=questao.id,
            topico=questao.topico,
            conceitos=questao.conceitos,
            dificuldade_alvo=questao.dificuldade,
            enunciado=questao.enunciado,
            alternativas=questao.alternativas,
            gabarito=questao.gabarito,
            descricoes_imagens=questao.descricoes_imagens,
            origem=OrigemItem.BANCO,
            referencias=[questao.id],
        )
