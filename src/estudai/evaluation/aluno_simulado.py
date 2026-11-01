"""Estudante simulado com conhecimento controlado externamente (Seção 5.2.2).

- Respostas a itens objetivos: modelo DINA (Deterministic Input, Noisy "And" gate):
  P(acerto) = 1 - slip se domina TODOS os conceitos do item, senão guess. O sorteio usa
  números aleatórios comuns por (persona, item, seed): entre pré e pós-teste, a resposta só
  muda se o estado de conhecimento mudou, o que reduz a variância do ganho medido.
- Diálogo com o Tutor: um LLM interpreta a persona, restrito ao mapa de conceitos.
- Transição lacuna -> dominado: decidida fora daqui, por juiz independente (experimento).
"""

import contextlib
import hashlib
import random
import time

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from estudai.agents.base import Agent, AgentOutputError, to_json
from estudai.domain.correcao import RespostaAluno
from estudai.domain.curriculum import Topico
from estudai.domain.mensagens import Papel
from estudai.domain.questoes import ItemObjetivo
from estudai.domain.tutoria import TurnoTutor
from estudai.evaluation.personas import Persona

SYSTEM = """Você interpreta um(a) estudante de ensino médio brasileiro(a) se preparando para o vestibular.
Persona: {descricao}
Conceitos que você DOMINA: {dominados}
Conceitos que você NÃO domina (lacunas): {lacunas}
Regras de interpretação:
- Responda como esse estudante, em português informal e curto (1 a 4 frases).
- Sobre lacunas, cometa erros típicos e plausíveis; só demonstre entendimento parcial se o
  tutor tiver acabado de explicar o conceito de forma clara nesta conversa.
- Nunca demonstre conhecimento além do seu perfil."""


class _Fala(BaseModel):
    texto: str


class _Resolucao(BaseModel):
    resolucao: str


class AlunoSimulado(Agent):
    papel = Papel.ALUNO_SIMULADO

    def __init__(self, persona: Persona, topico: Topico, llm: BaseChatModel | None = None, *, seed: int = 0):
        """Sem `llm`, o aluno ainda responde itens (DINA), mas sem resoluções/falas geradas."""
        super().__init__(llm)  # type: ignore[arg-type]
        self.persona = persona
        self.topico = topico
        self.dominados: set[str] = set(persona.dominados.get(topico, []))
        self.notificacoes: list[str] = []
        self._dialogo: list[str] = []
        self._seed = seed
        self._llm_disponivel = llm is not None
        self.tempo_llm_s = 0.0  # descontado da latência do sistema nos experimentos

    @property
    def lacunas(self) -> list[str]:
        return [c for c in self.topico.conceitos if c not in self.dominados]

    def aprender(self, conceitos: list[str]) -> None:
        self.dominados |= set(conceitos) & set(self.topico.conceitos)

    # --- respostas objetivas (sem LLM) ----------------------------------------------------

    def _rng(self, item_id: str) -> random.Random:
        chave = f"{self.persona.id}|{item_id}|{self._seed}".encode()
        return random.Random(int.from_bytes(hashlib.sha256(chave).digest()[:8], "big"))

    def p_acerto(self, conceitos: list[str]) -> float:
        p = self.persona
        if not conceitos:  # item sem rótulo: interpola pela fração dominada do tópico
            fracao = len(self.dominados) / len(self.topico.conceitos)
            return p.guess + (1 - p.slip - p.guess) * fracao
        return 1 - p.slip if set(conceitos) <= self.dominados else p.guess

    def escolher(self, item_id: str, item: ItemObjetivo) -> str:
        rng = self._rng(item_id)
        acerta = rng.random() < self.p_acerto(list(getattr(item, "conceitos", [])))
        erradas = [letra for letra in item.letras if letra != item.gabarito]
        return item.gabarito if acerta else rng.choice(erradas)

    def responder_bateria(self, itens: list) -> list[bool]:
        return [self.escolher(q.id, q) == q.gabarito for q in itens]

    # --- interface Learner ---------------------------------------------------------------

    def solve(self, item_id: str, item: ItemObjetivo) -> RespostaAluno:
        escolha = self.escolher(item_id, item)
        resolucao = ""
        if self._llm_disponivel:
            prompt = (
                f"{item.formatar()}\n\nVocê marcou a alternativa {escolha}. Escreva a resolução curta "
                "que você (como essa persona) teria feito para chegar nela, com os erros que cometeria."
            )
            with contextlib.suppress(AgentOutputError):
                resolucao = self._falar(prompt, _Resolucao).resolucao
        return RespostaAluno(item_id=item_id, escolha=escolha, resolucao=resolucao)

    def reply_to_tutor(self, turno: TurnoTutor) -> str:
        self._dialogo.append(f"TUTOR: {turno.explicacao}\nTUTOR (pergunta): {turno.pergunta_verificacao}")
        texto = "Não entendi muito bem, pode explicar de novo?"
        if self._llm_disponivel:
            prompt = "Conversa até aqui:\n" + "\n".join(self._dialogo[-8:]) + "\n\nResponda à última pergunta do tutor."
            with contextlib.suppress(AgentOutputError):
                texto = self._falar(prompt, _Fala).texto
        self._dialogo.append(f"ALUNO: {texto}")
        return texto

    def notify(self, mensagem: str) -> None:
        self.notificacoes.append(mensagem)

    def _falar(self, prompt: str, schema):
        inicio = time.perf_counter()
        try:
            return self._ask(self._system(), prompt, schema)
        finally:
            self.tempo_llm_s += time.perf_counter() - inicio

    def _system(self) -> str:
        return SYSTEM.format(
            descricao=self.persona.descricao,
            dominados=to_json(sorted(self.dominados)),
            lacunas=to_json(self.lacunas),
        )
