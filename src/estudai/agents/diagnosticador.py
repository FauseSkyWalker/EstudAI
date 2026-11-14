"""Agente Diagnosticador (modelo de fronteira).

- Sondagem inicial: escolhe perguntas do banco por cobertura gulosa de conceitos e para ao
  atingir a cobertura mínima ou o limite de perguntas (critério de término da Tabela 2).
- Estimativa/atualização: estima nível e lacunas por tópico a partir das evidências
  (respostas corrigidas). Se o LLM falhar, cai para uma estimativa determinística.
"""

import random
from collections import defaultdict

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from estudai.agents.base import Agent, AgentOutputError, to_json
from estudai.domain.correcao import ErroIdentificado
from estudai.domain.curriculum import Topico
from estudai.domain.mensagens import Papel
from estudai.domain.perfil import Diagnostico, NivelDominio, PerfilAluno
from estudai.domain.questoes import Dificuldade, Questao

SYSTEM = """Você é o Agente Diagnosticador de um sistema de tutoria adaptativa para vestibulares
brasileiros (Matemática e Física). A partir das evidências (respostas do aluno, corretas ou
não, com os erros identificados) e do perfil atual, estime o nível de domínio do aluno no
tópico (iniciante, intermediario, avancado), os conceitos dominados e as lacunas ativas.
- Use SOMENTE os rótulos de conceito fornecidos.
- Um conceito só é dominado com evidência de acerto; erros conceituais indicam lacuna.
- Evidência escassa deve resultar em confiança baixa, não em conclusões fortes.
"""


class Evidencia(BaseModel):
    item_id: str
    conceitos: list[str]
    dificuldade: Dificuldade
    correta: bool
    erros: list[ErroIdentificado] = Field(default_factory=list)


class Diagnosticador(Agent):
    papel = Papel.DIAGNOSTICADOR

    def __init__(
        self,
        model: BaseChatModel,
        *,
        max_perguntas: int = 6,
        cobertura_minima: float = 0.8,
        **kw,
    ):
        super().__init__(model, **kw)
        self._max_perguntas = max_perguntas
        self._cobertura_minima = cobertura_minima

    def selecionar_sondagem(self, topico: Topico, banco: list[Questao], seed: int = 0) -> list[Questao]:
        """Seleção gulosa: cada pergunta deve cobrir ao menos um conceito ainda não coberto."""
        pool = [q for q in banco if q.topico == topico and q.conceitos]
        random.Random(seed).shuffle(pool)
        pool.sort(key=lambda q: q.dificuldade != Dificuldade.MEDIA)  # média primeiro (estável)
        alvo = set(topico.conceitos)
        cobertos: set[str] = set()
        escolhidas: list[Questao] = []
        while len(escolhidas) < self._max_perguntas and len(cobertos) / len(alvo) < self._cobertura_minima:
            melhor = max(pool, key=lambda q: len(set(q.conceitos) - cobertos), default=None)
            if melhor is None or not set(melhor.conceitos) - cobertos:
                break
            escolhidas.append(melhor)
            cobertos |= set(melhor.conceitos)
            pool.remove(melhor)
        return escolhidas

    def estimar(self, perfil: PerfilAluno, topico: Topico, evidencias: list[Evidencia]) -> Diagnostico:
        prompt = (
            f"Tópico: {topico.value}\nConceitos possíveis: {list(topico.conceitos)}\n\n"
            f"Perfil atual:\n{to_json(perfil.visao(topico))}\n\n"
            f"Resumo por conceito (acertos/total):\n{to_json(_acertos_por_conceito(evidencias))}\n\n"
            f"Evidências:\n{to_json([e.model_dump(mode='json') for e in evidencias])}"
        )
        try:
            diagnostico = self._ask(SYSTEM, prompt, Diagnostico)
        except AgentOutputError:
            return estimativa_deterministica(topico, evidencias)
        return diagnostico.model_copy(update={"topico": topico})


def _contagem(evidencias: list[Evidencia]) -> dict[str, list[int]]:
    """conceito -> [acertos, total]"""
    contagem: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for e in evidencias:
        for c in e.conceitos:
            contagem[c][0] += int(e.correta)
            contagem[c][1] += 1
    return contagem


def _acertos_por_conceito(evidencias: list[Evidencia]) -> dict[str, str]:
    return {c: f"{a}/{t}" for c, (a, t) in _contagem(evidencias).items()}


def estimativa_deterministica(topico: Topico, evidencias: list[Evidencia]) -> Diagnostico:
    """Fallback: conceito dominado se a maioria das evidências dele está correta."""
    contagem = _contagem(evidencias)
    dominados = [c for c, (a, t) in contagem.items() if a / t > 0.5]
    lacunas = [c for c, (a, t) in contagem.items() if a / t <= 0.5]
    return Diagnostico(
        topico=topico,
        nivel=NivelDominio.da_fracao(len(dominados) / len(topico.conceitos)),
        conceitos_dominados=dominados,
        lacunas=lacunas,
        justificativa="Estimativa determinística (fallback).",
        confianca=min(0.9, 0.15 * len(evidencias)),
    )
