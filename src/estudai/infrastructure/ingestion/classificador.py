"""Rotulagem de questões por LLM (modelo econômico): disciplina, tópico, conceitos e dificuldade."""

from typing import Literal

from pydantic import BaseModel, Field

from estudai.agents.base import Agent
from estudai.domain.curriculum import CONCEITOS
from estudai.domain.mensagens import Papel
from estudai.domain.questoes import Dificuldade
from estudai.infrastructure.ingestion.sources import QuestaoBruta

_CATALOGO = "\n".join(f"- {t.value}: {', '.join(c)}" for t, c in CONCEITOS.items())

SYSTEM = f"""Você classifica questões de vestibular brasileiro. Determine:
- disciplina: "matematica", "fisica" ou "outra";
- topico: um de "funcoes", "probabilidade", "cinematica", "dinamica", ou "outro" se a questão
  não pertencer predominantemente a nenhum deles;
- conceitos: rótulos exigidos para resolver, SOMENTE do catálogo do tópico escolhido;
- dificuldade: "facil", "media" ou "dificil" para um aluno de ensino médio.
Catálogo de conceitos por tópico:
{_CATALOGO}"""


class Rotulo(BaseModel):
    disciplina: Literal["matematica", "fisica", "outra"]
    topico: Literal["funcoes", "probabilidade", "cinematica", "dinamica", "outro"]
    conceitos: list[str] = Field(default_factory=list)
    dificuldade: Dificuldade = Dificuldade.MEDIA


class ClassificadorQuestoes(Agent):
    papel = Papel.CLASSIFICADOR

    def classificar(self, questao: QuestaoBruta) -> Rotulo:
        dica = (
            f"(metadado da base: disciplina={questao.disciplina_indicada.value})\n"
            if questao.disciplina_indicada
            else ""
        )
        alternativas = "\n".join(f"- {a}" for a in questao.alternativas)
        return self._ask(SYSTEM, f"{dica}{questao.enunciado[:4000]}\n\nAlternativas:\n{alternativas}", Rotulo)
