"""Agente de Resumo (modelo econômico): material de revisão e relatório da prática."""

from estudai.agents.base import Agent, to_json
from estudai.domain.correcao import Correcao, RelatorioPratica
from estudai.domain.material import PacoteMaterial
from estudai.domain.mensagens import Papel
from estudai.domain.tutoria import ResumoEstudo, TranscricaoTutoria

SYSTEM = """Você é o Agente de Resumo de um sistema de tutoria para vestibulares brasileiros.
Produza sínteses fiéis ao que foi efetivamente ensinado/praticado, sem acrescentar conteúdo novo.
Seja conciso: listas curtas, fórmulas em notação simples (ex.: v = v0 + a*t)."""


class AgenteResumo(Agent):
    papel = Papel.RESUMO

    def resumir_tutoria(self, transcricao: TranscricaoTutoria, material: PacoteMaterial) -> ResumoEstudo:
        prompt = (
            f"Objetivo da sessão: {transcricao.objetivo.descricao}\n\n"
            f"Transcrição da tutoria:\n{transcricao.como_texto()}\n\n"
            f"Material-fonte usado:\n{material.como_contexto(max_chars=3000)}\n\n"
            "Gere o resumo de revisão para o aluno."
        )
        return self._ask(SYSTEM, prompt, ResumoEstudo)

    def resumir_pratica(self, correcoes: list[Correcao], perfil_antes: dict, perfil_depois: dict) -> RelatorioPratica:
        prompt = (
            f"Correções dos exercícios:\n{to_json([c.model_dump(mode='json') for c in correcoes])}\n\n"
            f"Perfil antes da prática:\n{to_json(perfil_antes)}\n\n"
            f"Perfil depois da prática:\n{to_json(perfil_depois)}\n\n"
            "Gere o relatório de desempenho para o aluno."
        )
        return self._ask(SYSTEM, prompt, RelatorioPratica)
