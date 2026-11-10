"""Agente Tutor por Disciplina (modelo de fronteira): uma instância para Matemática e outra para Física."""

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel

from estudai.agents.base import Agent, to_json
from estudai.domain.curriculum import Disciplina
from estudai.domain.latex import ESPECIFICACAO_LATEX, para_inline
from estudai.domain.material import PacoteMaterial
from estudai.domain.mensagens import Papel
from estudai.domain.planejamento import ObjetivoSessao
from estudai.domain.tutoria import TranscricaoTutoria, TurnoTutor

_ORIENTACOES = {
    Disciplina.MATEMATICA: (
        "Explique a ideia antes do procedimento, mostre um exemplo numérico curto e destaque "
        "armadilhas comuns de vestibular (sinais, domínio, casos particulares)."
    ),
    Disciplina.FISICA: (
        "Parta do fenômeno físico, explicite grandezas e unidades, indique o referencial e só "
        "então a equação; verifique a coerência dimensional e a ordem de grandeza."
    ),
}

SYSTEM = """Você é o Agente Tutor de {disciplina} de um sistema de tutoria para vestibulares brasileiros.
Conduza a sessão em rodadas curtas, em português claro, adaptadas ao nível do aluno.
{orientacao}
Regras:
- Ensine UM conceito-alvo por rodada e termine com UMA pergunta de verificação.
- Fundamente-se no material fornecido e cite os ids entre colchetes, ex.: [wb-123]. Não invente fatos.
- Avalie a resposta anterior do aluno antes de seguir; se errada, reexplique de outro jeito.
- Inclua em `conceitos_validados` apenas conceitos que o aluno demonstrou entender respondendo
  corretamente a uma pergunta de verificação nesta sessão.
- Marque `objetivo_atingido` somente quando todos os conceitos-alvo estiverem validados.
- Use `sessoes_anteriores` do perfil (memória das sessões passadas) para dar continuidade: na
  primeira rodada, retome em uma frase o que já foi visto; priorize os `proximos_passos` e os
  `erros_comuns`/`conceitos_com_erro` que se repetem; não reexplique do zero o que o aluno já
  dominou. Sem sessões anteriores, comece normalmente.
"""


class Tutor(Agent):
    papel = Papel.TUTOR

    def __init__(
        self, model: BaseChatModel, disciplina: Disciplina, *, usage_cb: BaseCallbackHandler | None = None, **kw
    ):
        super().__init__(model, usage_cb, **kw)
        self.disciplina = disciplina
        self._system = (
            SYSTEM.format(disciplina=disciplina.value, orientacao=_ORIENTACOES[disciplina]) + ESPECIFICACAO_LATEX
        )

    def proximo_turno(
        self,
        perfil: dict,
        objetivo: ObjetivoSessao,
        material: PacoteMaterial,
        transcricao: TranscricaoTutoria,
    ) -> TurnoTutor:
        historico = transcricao.como_texto() or "(início da sessão)"
        prompt = (
            f"Objetivo da sessão:\n{to_json(objetivo)}\n\n"
            f"Perfil do aluno:\n{to_json(perfil)}\n\n"
            f"Material de apoio:\n{material.como_contexto()}\n\n"
            f"Conversa até aqui:\n{historico}\n\n"
            f"Conceitos já validados: {transcricao.conceitos_validados}\n"
            "Produza a próxima rodada."
        )
        turno = self._ask(self._system, prompt, TurnoTutor)
        return turno.model_copy(
            update={
                "avaliacao_resposta_anterior": (
                    para_inline(turno.avaliacao_resposta_anterior) if turno.avaliacao_resposta_anterior else None
                ),
                "explicacao": para_inline(turno.explicacao),
                "pergunta_verificacao": para_inline(turno.pergunta_verificacao),
            }
        )
