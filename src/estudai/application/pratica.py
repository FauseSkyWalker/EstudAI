"""Estratégias de fornecimento de exercícios para a macrofase de prática.

- `GeneratedExerciseProvider`: Gerador + validação por consenso (condições C1 e C2).
- `BankExerciseProvider`: somente questões reais do banco (condição C3, grupo de controle).
"""

import random
from typing import Protocol

from pydantic import BaseModel, Field

from estudai.agents.base import AgentOutputError
from estudai.agents.gerador import GeradorExercicios
from estudai.application.validacao import ResultadoValidacao, ValidadorExercicios
from estudai.domain.curriculum import Topico
from estudai.domain.perfil import PerfilAluno
from estudai.domain.planejamento import planejar_pratica
from estudai.domain.questoes import Exercicio, Particao
from estudai.ports.knowledge import QuestionBank


class ItemDescartado(BaseModel):
    exercicio: Exercicio | None
    motivo: str
    validacao: ResultadoValidacao | None = None


class ResultadoFornecimento(BaseModel):
    exercicios: list[Exercicio] = Field(default_factory=list)
    validacoes: dict[str, ResultadoValidacao] = Field(default_factory=dict)
    descartados: list[ItemDescartado] = Field(default_factory=list)
    falhas_geracao: int = 0

    @property
    def taxa_descarte(self) -> float:
        validados = len(self.exercicios) + len(self.descartados)
        return len(self.descartados) / validados if validados else 0.0


class ExerciseProvider(Protocol):
    def fornecer(
        self, perfil: PerfilAluno, topico: Topico, n: int, excluir: frozenset[str] = frozenset()
    ) -> ResultadoFornecimento: ...


class GeneratedExerciseProvider:
    def __init__(self, gerador: GeradorExercicios, validador: ValidadorExercicios, *, tentativas_por_item: int = 2):
        self._gerador = gerador
        self._validador = validador
        self._tentativas = tentativas_por_item

    def fornecer(
        self, perfil: PerfilAluno, topico: Topico, n: int, excluir: frozenset[str] = frozenset()
    ) -> ResultadoFornecimento:
        resultado = ResultadoFornecimento()
        visao = perfil.visao(topico)
        for plano in planejar_pratica(perfil, topico, n):
            for _ in range(self._tentativas):
                evitar = [e.enunciado for e in resultado.exercicios]
                try:
                    exercicio = self._gerador.gerar(visao, topico, plano, evitar)
                except AgentOutputError as exc:
                    resultado.falhas_geracao += 1
                    resultado.descartados.append(ItemDescartado(exercicio=None, motivo=f"geração falhou: {exc}"))
                    continue
                validacao = self._validador.validar(exercicio)
                if validacao.aprovado:
                    resultado.exercicios.append(exercicio)
                    resultado.validacoes[exercicio.id] = validacao
                    break
                resultado.descartados.append(
                    ItemDescartado(exercicio=exercicio, motivo=validacao.motivo, validacao=validacao)
                )
        return resultado


class BankExerciseProvider:
    def __init__(self, banco: QuestionBank, *, seed: int = 0) -> None:
        self._banco = banco
        self._rng = random.Random(seed)

    def fornecer(
        self, perfil: PerfilAluno, topico: Topico, n: int, excluir: frozenset[str] = frozenset()
    ) -> ResultadoFornecimento:
        disponiveis = [q for q in self._banco.find(topico, Particao.BANCO) if q.id not in excluir]
        self._rng.shuffle(disponiveis)
        escolhidas = []
        for plano in planejar_pratica(perfil, topico, n):
            candidatas = [q for q in disponiveis if plano.conceito in q.conceitos] or disponiveis
            if not candidatas:
                break
            # prefere a dificuldade planejada, sem perder a aleatoriedade do embaralhamento
            q = min(candidatas, key=lambda q: q.dificuldade != plano.dificuldade)
            escolhidas.append(q)
            disponiveis.remove(q)
        return ResultadoFornecimento(exercicios=[Exercicio.da_questao(q) for q in escolhidas])
