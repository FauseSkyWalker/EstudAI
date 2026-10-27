"""Resolutor independente do painel de consenso: resolve o item SEM ver o gabarito."""

from pydantic import BaseModel, Field

from estudai.agents.base import Agent
from estudai.domain.mensagens import Papel
from estudai.domain.questoes import ItemObjetivo

SYSTEM = """Você é um resolutor independente de questões objetivas de Matemática e Física de
vestibulares brasileiros. Resolva com cuidado, passo a passo, e escolha UMA alternativa.
Sinalize se mais de uma alternativa estiver correta ou se o enunciado for ambíguo/insolúvel."""


class VotoResolutor(BaseModel):
    raciocinio: str = Field(description="Resolução curta.")
    resposta: str = Field(description="Letra da alternativa escolhida.")
    mais_de_uma_correta: bool = False
    enunciado_problematico: bool = False


class Resolutor(Agent):
    papel = Papel.PAINEL_RESOLUTORES

    def resolver(self, item: ItemObjetivo) -> VotoResolutor:
        enunciado_sem_gabarito = item.formatar(com_gabarito=False)
        voto = self._ask(SYSTEM, enunciado_sem_gabarito, VotoResolutor)
        return voto.model_copy(update={"resposta": voto.resposta.strip().upper()[:1]})
