"""Juízes LLM da avaliação com prompt fixos

Recomenda-se usar juízes de fornecedor diferente do sistema avaliado e mais de
um juiz, para medir a concordância (κ de Fleiss) no relatório.
"""

from typing import Annotated

from pydantic import BaseModel, Field

from estudai.agents.base import Agent, to_json
from estudai.domain.material import PacoteMaterial
from estudai.domain.mensagens import Papel
from estudai.domain.questoes import Exercicio
from estudai.domain.tutoria import TranscricaoTutoria

Nota = Annotated[int, Field(ge=1, le=5)]


class VeredictoTransicao(BaseModel):
    explicou_explicitamente: bool
    validou_entendimento: bool
    justificativa: str

    @property
    def aprendeu(self) -> bool:
        return self.explicou_explicitamente and self.validou_entendimento


class NotasTutoria(BaseModel):
    """Rubrica de 4 dimensões adaptada de B. Zhao et al. (2026), escala 1–5."""

    fidelidade_material: Nota
    personalizacao: Nota
    aplicabilidade: Nota
    profundidade_logica: Nota
    justificativa: str

    @property
    def media(self) -> float:
        return (self.fidelidade_material + self.personalizacao + self.aplicabilidade + self.profundidade_logica) / 4


class AvaliacaoItem(BaseModel):
    """Qualidade textual e aderência curricular (W. Fu et al., 2024)."""

    vinculo_topico: bool
    clareza: Nota
    plausibilidade_distratores: Nota
    sem_pistas_gramaticais: bool
    defeito_critico: bool
    justificativa: str


class JuizTransicao(Agent):
    papel = Papel.JUIZ
    SYSTEM = """Você é um avaliador independente de sessões de tutoria. Dado o diálogo e um
conceito, decida (1) se o tutor EXPLICOU explicitamente esse conceito e (2) se o tutor
VALIDOU o entendimento do aluno (fez pergunta de verificação sobre ele e o aluno respondeu
corretamente). Seja rigoroso: menção superficial não conta como explicação."""

    def avaliar(self, transcricao: TranscricaoTutoria, conceito: str) -> VeredictoTransicao:
        prompt = f"Conceito: {conceito}\n\nDiálogo:\n{transcricao.como_texto()}"
        return self._ask(self.SYSTEM, prompt, VeredictoTransicao)


class JuizTutoria(Agent):
    papel = Papel.JUIZ
    SYSTEM = """Você avalia sessões de tutoria de Matemática/Física para vestibular, de 1 a 5, em:
- fidelidade_material: a explicação é correta e coerente com o material-fonte fornecido;
- personalizacao: adapta-se ao nível, às lacunas e às respostas do aluno;
- aplicabilidade: conecta com resolução de questões de vestibular;
- profundidade_logica: o raciocínio é encadeado, justifica passos e trata erros conceituais.
Use a escala inteira; 3 = adequado, 5 = excelente sem ressalvas."""

    def avaliar(self, transcricao: TranscricaoTutoria, material: PacoteMaterial, perfil: dict) -> NotasTutoria:
        prompt = (
            f"Perfil do aluno no início:\n{to_json(perfil)}\n\n"
            f"Objetivo: {transcricao.objetivo.descricao}\n\n"
            f"Material-fonte:\n{material.como_contexto(max_chars=4000)}\n\n"
            f"Diálogo:\n{transcricao.como_texto()}"
        )
        return self._ask(self.SYSTEM, prompt, NotasTutoria)


class JuizItem(Agent):
    papel = Papel.JUIZ
    SYSTEM = """Você avalia questões objetivas de vestibular (Matemática/Física). Verifique:
vínculo com o tópico/conceito-alvo; clareza do enunciado (1–5); plausibilidade dos distratores
(1–5); ausência de pistas gramaticais/de tamanho; e se há defeito crítico (gabarito errado,
mais de uma correta, dados faltantes, enunciado insolúvel)."""

    def avaliar(self, item: Exercicio) -> AvaliacaoItem:
        prompt = (
            f"Tópico: {item.topico.value}; conceitos-alvo: {item.conceitos}; "
            f"dificuldade-alvo: {item.dificuldade_alvo.value}\n\n{item.formatar(com_gabarito=True)}"
        )
        return self._ask(self.SYSTEM, prompt, AvaliacaoItem)
