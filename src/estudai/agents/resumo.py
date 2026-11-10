"""Summary agent, which summarizes study material and the overall cycle"""

from estudai.agents.base import Agent, to_json
from estudai.domain.correcao import Correcao, RelatorioPratica
from estudai.domain.latex import ESPECIFICACAO_LATEX, para_inline
from estudai.domain.material import PacoteMaterial
from estudai.domain.mensagens import Papel
from estudai.domain.tutoria import ResumoEstudo, TranscricaoTutoria

SYSTEM = (
    """Você é o Agente de Resumo de um sistema de tutoria para vestibulares brasileiros.
Produza sínteses fiéis ao que foi efetivamente ensinado/praticado, sem acrescentar conteúdo novo.
Seja conciso: listas curtas e fórmulas matemáticas em notação LaTeX. Mantenha a postura didática."""
    + ESPECIFICACAO_LATEX
)


class AgenteResumo(Agent):
    papel = Papel.RESUMO

    def resumir_tutoria(self, transcricao: TranscricaoTutoria, material: PacoteMaterial) -> ResumoEstudo:
        prompt = (
            f"Objetivo da sessão: {transcricao.objetivo.descricao}\n\n"
            f"Transcrição da tutoria:\n{transcricao.como_texto()}\n\n"
            f"Material-fonte usado:\n{material.como_contexto(max_chars=3000)}\n\n"
            "Gere o resumo de revisão para o aluno."
        )
        resumo = self._ask(SYSTEM, prompt, ResumoEstudo)
        return resumo.model_copy(
            update={
                "titulo": para_inline(resumo.titulo),
                "pontos_chave": [para_inline(p) for p in resumo.pontos_chave],
                "formulas": [para_inline(f) for f in resumo.formulas],
                "erros_comuns": [para_inline(e) for e in resumo.erros_comuns],
                "proximos_passos": [para_inline(p) for p in resumo.proximos_passos],
            }
        )

    def resumir_pratica(self, correcoes: list[Correcao], perfil_antes: dict, perfil_depois: dict) -> RelatorioPratica:
        prompt = (
            f"Correções dos exercícios:\n{to_json([c.model_dump(mode='json') for c in correcoes])}\n\n"
            f"Perfil antes da prática:\n{to_json(perfil_antes)}\n\n"
            f"Perfil depois da prática:\n{to_json(perfil_depois)}\n\n"
            "Gere o relatório de desempenho para o aluno."
        )
        relatorio = self._ask(SYSTEM, prompt, RelatorioPratica)
        return relatorio.model_copy(
            update={
                "desempenho": para_inline(relatorio.desempenho),
                "pontos_fortes": [para_inline(p) for p in relatorio.pontos_fortes],
                "pontos_a_melhorar": [para_inline(p) for p in relatorio.pontos_a_melhorar],
                "recomendacoes": [para_inline(r) for r in relatorio.recomendacoes],
            }
        )
