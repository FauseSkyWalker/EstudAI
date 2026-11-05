"""Rotulagem e normalização de questões por LLM (modelo econômico).

Uma única chamada faz as duas coisas: classifica (disciplina/tópico/conceitos/dificuldade) e,
para questões dentro do escopo, devolve o texto com a matemática em LaTeX e os artefatos de
extração corrigidos. Questões fora do escopo não pagam o custo de devolver o texto.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from estudai.agents.base import Agent
from estudai.domain.curriculum import CONCEITOS, Topico
from estudai.domain.latex import latex_valido, para_inline
from estudai.domain.mensagens import Papel
from estudai.domain.questoes import Dificuldade
from estudai.infrastructure.ingestion.normalizacao import texto_invalido
from estudai.infrastructure.ingestion.sources import QuestaoBruta

_CATALOGO = "\n".join(f"- {t.value}: {', '.join(c)}" for t, c in CONCEITOS.items())

_SYSTEM = r"""Você classifica e normaliza questões de vestibular brasileiro.

Classificação:
- disciplina: "matematica", "fisica" ou "outra"
- topico: um de "funcoes", "probabilidade", "cinematica", "dinamica", ou "outro" se o FOCO PRINCIPAL
  não pertencer predominantemente a nenhum deles
- conceitos: rótulos exigidos para resolver, SOMENTE do catálogo do tópico escolhido
- dificuldade: "facil", "media" ou "dificil" para um aluno de ensino médio.

Normalização (somente se a questão for de matemática ou física e o tópico for um dos quatro):
- Reescreva `enunciado` e `alternativas` convertendo TODA a matemática para LaTeX inline $...$.
- IMPORTANTE: nos campos de texto use exclusivamente $...$ (cifrão), nunca \(...\), \[...\] ou $$...$$.
- Cifrões de moeda (R$) NÃO são matemática: preserve "R$ 5,00" literalmente, sem LaTeX.
- Use \frac, \sqrt, ^{}, _{}, \mathbb{R}, \le, \ge, \times, \pi, etc.
- Corrija agrupamentos/parênteses, frações, raízes, expoentes, índices, conjuntos e unidades.
- Corrija artefatos de extração: quebras de linha no meio de palavras, hífens, espaços duplos,
  prefixos "Descrição da imagem:" e títulos markdown.
- Preserve o português, o sentido, a ordem e o número de alternativas. Não invente dados.
- Se a questão for "outra"/"outro", deixe `enunciado` e `alternativas` nulos.

Exemplos de normalização:
- "D_f" -> "$D_{f}$"; "R" (conjunto dos reais) -> "$\mathbb{R}$"
- "sqrt((x^3 + 2x^2 - 4x - 8)/x-2)" -> "$\sqrt{\frac{x^{3} + 2x^{2} - 4x - 8}{x-2}}$"
- "10^3" -> "$10^{3}$"

Responda em JSON.
"""
SYSTEM = f"{_SYSTEM}\nCatálogo de conceitos por tópico:\n{_CATALOGO}"


class Rotulo(BaseModel):
    disciplina: Literal["matematica", "fisica", "outra"]
    topico: Literal["funcoes", "probabilidade", "cinematica", "dinamica", "outro"]
    conceitos: list[str] = Field(default_factory=list)
    dificuldade: Dificuldade = Dificuldade.MEDIA
    enunciado: str | None = Field(default=None, description="Enunciado normalizado em LaTeX. Nulo para fora do escopo")
    alternativas: list[str] | None = Field(
        default=None, description="Alternativas normalizadas em LaTex. Nulo fora do escopo"
    )

    @model_validator(mode="after")
    def check_conceitos_match_topico(self):
        if self.topico == "outro":
            return self

        topico_enum = Topico(self.topico)

        validos = CONCEITOS.get(topico_enum, [])
        invalidos = [c for c in self.conceitos if c not in validos]

        if invalidos:
            raise ValueError(
                f"Os conceitos {invalidos} não pertencem ao tópico '{self.topico}'. "
                f"Para este tópico, use apenas: {', '.join(validos)}"
            )

        return self


class ClassificadorQuestoes(Agent):
    papel = Papel.CLASSIFICADOR

    def classificar(self, questao: QuestaoBruta) -> Rotulo:
        dica = (
            f"(metadado da base: disciplina={questao.disciplina_indicada.value})\n"
            if questao.disciplina_indicada
            else ""
        )
        alternativas = "\n".join(f"- {a}" for a in questao.alternativas)
        rotulo = self._ask(SYSTEM, f"{dica}{questao.enunciado[:4000]}\n\nAlternativas:\n{alternativas}", Rotulo)

        if rotulo.disciplina != "outra" and rotulo.topico != "outro":
            normalizadas = rotulo.alternativas or questao.alternativas
            if len(normalizadas) != len(questao.alternativas):
                normalizadas = questao.alternativas
            enunciado = para_inline(rotulo.enunciado or questao.enunciado)
            alternativas = [para_inline(a) for a in normalizadas]
            if (
                texto_invalido(enunciado)
                or any(texto_invalido(a) for a in alternativas)
                or not latex_valido(enunciado)
                or any(not latex_valido(a) for a in alternativas)
            ):
                enunciado, alternativas = questao.enunciado, questao.alternativas  # não grava saída suspeita
            rotulo.enunciado, rotulo.alternativas = enunciado, alternativas

        return rotulo
