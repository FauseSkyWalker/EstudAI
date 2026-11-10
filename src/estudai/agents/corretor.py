"""Agente Corretor (modelo de fronteira): avalia cada item individualmente.

A corretude objetiva é determinística (escolha == gabarito). O LLM só diagnostica o
raciocínio: tipo de erro (conceitual/numérico/interpretação), conceito envolvido, crédito
parcial e se o gabarito parece suspeito (o que dispara o painel de resolutores).
"""

from estudai.agents.base import Agent
from estudai.domain.correcao import AnaliseCorrecao, Correcao, RespostaAluno, acertou
from estudai.domain.latex import ESPECIFICACAO_LATEX, para_inline
from estudai.domain.mensagens import Papel
from estudai.domain.questoes import Exercicio

SYSTEM = (
    """Você é o Agente Corretor de um sistema de tutoria para vestibulares brasileiros
(Matemática e Física). Analise a resolução do aluno para UM item.
- Classifique cada erro como "conceitual", "numerico" ou "interpretacao" e indique o conceito
  (use os rótulos do item).
- Atribua `nota` de 0 a 1 ao raciocínio (crédito parcial é permitido mesmo com alternativa errada).
- Se a sua própria resolução contradiz o gabarito informado, marque `gabarito_suspeito`.
- Feedback curto, construtivo e em português."""
    + ESPECIFICACAO_LATEX
)


class Corretor(Agent):
    papel = Papel.CORRETOR

    def corrigir(self, item: Exercicio, resposta: RespostaAluno) -> Correcao:
        correta = acertou(resposta, item.gabarito)
        base = {"item_id": item.id, "escolha": resposta.escolha, "gabarito": item.gabarito, "correta": correta}
        if correta and not resposta.resolucao.strip():  # Correct response, avoid an LLM call
            return Correcao(**base, nota=1.0, feedback="Resposta correta.")

        prompt = (
            f"Item (conceitos: {item.conceitos}):\n{item.formatar(com_gabarito=True)}\n\n"
            f"Resolução de referência:\n{item.resolucao or '(não disponível)'}\n\n"
            f"Resposta do aluno: {resposta.escolha or '(em branco)'} "
            f"({'correta' if correta else 'incorreta'})\n"
            f"Resolução do aluno:\n{resposta.resolucao or '(não apresentada)'}"
        )
        analise = self._ask(SYSTEM, prompt, AnaliseCorrecao)
        analise = analise.model_copy(
            update={
                "feedback": para_inline(analise.feedback),
                "erros": [erro.model_copy(update={"descricao": para_inline(erro.descricao)}) for erro in analise.erros],
            }
        )
        return Correcao(**base, **analise.model_dump())
