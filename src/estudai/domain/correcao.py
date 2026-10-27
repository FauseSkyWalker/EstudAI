"""Macrofase de prática: respostas do aluno, correções e relatório de desempenho."""

from pydantic import BaseModel, Field

from estudai.domain.perfil import TipoErro


class RespostaAluno(BaseModel):
    item_id: str
    escolha: str | None = None  # letra; None = em branco
    resolucao: str = ""


class ErroIdentificado(BaseModel):
    tipo: TipoErro
    conceito: str | None = None
    descricao: str


class AnaliseCorrecao(BaseModel):
    """Parte da correção produzida pelo LLM. A corretude objetiva NÃO vem do LLM."""

    nota: float = Field(ge=0.0, le=1.0, description="Crédito pelo raciocínio, de 0 a 1.")
    erros: list[ErroIdentificado] = Field(default_factory=list)
    feedback: str
    gabarito_suspeito: bool = Field(default=False, description="True se o gabarito informado parece incorreto.")


class Correcao(AnaliseCorrecao):
    item_id: str
    escolha: str | None
    gabarito: str
    correta: bool  # comparação determinística escolha == gabarito


class RelatorioPratica(BaseModel):
    """Saída estruturada do Agente de Resumo ao fim da prática."""

    desempenho: str
    pontos_fortes: list[str] = Field(default_factory=list)
    pontos_a_melhorar: list[str] = Field(default_factory=list)
    recomendacoes: list[str] = Field(default_factory=list)
