"""Os 3 perfis simulados fixos (Seção 5.3), com nível conhecido por tópico.

O estado de conhecimento é um mapa externo de conceitos dominados: é ele, e não o LLM que
interpreta o aluno, que define o "nível verdadeiro" (evita circularidade, Seção 5.2.2).
"""

from pydantic import BaseModel, Field

from estudai.domain.curriculum import CONCEITOS, Topico
from estudai.domain.perfil import NivelDominio


class Persona(BaseModel):
    id: str
    descricao: str
    dominados: dict[Topico, list[str]]
    slip: float = Field(default=0.1, description="P(errar | domina todos os conceitos do item)")
    guess: float = Field(default=0.2, description="P(acertar | falta algum conceito) ~ chute em 5 alternativas")

    def nivel_verdadeiro(self, topico: Topico, dominados: set[str] | None = None) -> NivelDominio:
        dominados = set(self.dominados.get(topico, [])) if dominados is None else dominados
        return NivelDominio.da_fracao(len(dominados & set(topico.conceitos)) / len(topico.conceitos))


_T = Topico
PERSONAS: tuple[Persona, ...] = (
    Persona(
        id="p1_iniciante",
        descricao="Aluna de escola pública, base fraca em exatas; confunde fórmulas e desiste cedo.",
        dominados={
            _T.FUNCOES: ["dominio_imagem"],
            _T.PROBABILIDADE: ["principio_contagem"],
            _T.CINEMATICA: ["movimento_uniforme"],
            _T.DINAMICA: [],
        },
        slip=0.15,
    ),
    Persona(
        id="p2_misto",
        descricao="Bom em Matemática, inseguro em Física; aplica fórmulas sem interpretar o fenômeno.",
        dominados={
            _T.FUNCOES: list(CONCEITOS[_T.FUNCOES][:5]),
            _T.PROBABILIDADE: list(CONCEITOS[_T.PROBABILIDADE][:4]),
            _T.CINEMATICA: ["movimento_uniforme"],
            _T.DINAMICA: ["leis_de_newton"],
        },
    ),
    Persona(
        id="p3_intermediario",
        descricao="Intermediário com lacunas pontuais; erra por desatenção em contas.",
        dominados={
            _T.FUNCOES: ["dominio_imagem", "funcao_afim", "funcao_quadratica"],
            _T.PROBABILIDADE: ["principio_contagem", "permutacoes_combinacoes"],
            _T.CINEMATICA: ["movimento_uniforme", "movimento_uniformemente_variado", "queda_livre"],
            _T.DINAMICA: ["leis_de_newton", "atrito"],
        },
        slip=0.12,
    ),
)
