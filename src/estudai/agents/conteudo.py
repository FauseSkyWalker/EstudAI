"""Agente de Conteúdo (modelo econômico, LangChain + ferramentas): recupera e ranqueia material de apoio.

Validação de fidelidade: o agente apenas *seleciona* ids de documentos que as ferramentas
de fato retornaram; o texto entregue é sempre o original da base. Ids inventados são
descartados, então a saída é fiel às fontes por construção.
"""

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from estudai.agents.base import Agent, AgentOutputError
from estudai.agents.tools import search_tool
from estudai.domain.curriculum import Topico
from estudai.domain.material import PacoteMaterial, TipoDocumento, TrechoRecuperado
from estudai.domain.mensagens import Papel
from estudai.ports.knowledge import KnowledgeBase, SearchHit, SearchMode

SYSTEM = """Você é o Agente de Conteúdo de um sistema de tutoria de Matemática e Física para
vestibulares brasileiros. Sua função é buscar na base de conhecimento os trechos didáticos e
as questões de referência mais úteis para o pedido recebido.
- Faça buscas com consultas curtas e específicas; varie a formulação se os resultados forem fracos.
- Prefira busca híbrida; use a textual para fórmulas/termos exatos e a semântica para ideias.
- Selecione apenas ids que apareceram nos resultados das buscas, do mais ao menos relevante.
"""


class SelecaoConteudo(BaseModel):
    trechos: list[str] = Field(default_factory=list, description="ids de material, em ordem de relevância")
    questoes_referencia: list[str] = Field(default_factory=list, description="ids de questões")
    justificativa: str = ""


class AgenteConteudo(Agent):
    papel = Papel.CONTEUDO

    def __init__(
        self,
        model: BaseChatModel,
        kb: KnowledgeBase,
        *,
        usage_cb: BaseCallbackHandler | None = None,
        k: int = 4,
        min_score: float = 0.2,
        max_steps: int = 4,
        **kw,
    ) -> None:
        super().__init__(model, usage_cb, **kw)
        self._kb = kb
        self._k = k
        self._min_score = min_score
        self._max_steps = max_steps

    def buscar(self, pedido: str, topico: Topico | None = None, incluir_questoes: bool = True) -> PacoteMaterial:
        candidates: dict[str, SearchHit] = {}
        tools = [
            search_tool(SearchMode.HIBRIDA, topico, candidates, self._kb, self._k, self._min_score),
            search_tool(SearchMode.SEMANTICA, topico, candidates, self._kb, self._k, self._min_score),
            search_tool(SearchMode.TEXTUAL, topico, candidates, self._kb, self._k, self._min_score),
        ]

        prompt = (
            f"Pedido: {pedido}\nTópico: {topico.value if topico else 'qualquer'}\n"
            f"Selecione até {self._k} trechos de material"
            + (f" e até {self._k} questões de referência." if incluir_questoes else ".")
        )

        try:
            selecao = self._run_tool_agent(prompt, tools, SYSTEM, SelecaoConteudo, self._max_steps)
        except AgentOutputError:
            selecao = SelecaoConteudo()

        # model didnt search, search manually
        if not candidates:
            for hit in self._kb.search(pedido, SearchMode.HIBRIDA, k=2 * self._k, topico=topico):
                if hit.score >= self._min_score:
                    candidates[hit.document.id] = hit

        return PacoteMaterial(
            pedido=pedido,
            trechos=self._montar(selecao.trechos, candidates, TipoDocumento.MATERIAL),
            questoes_referencia=(
                self._montar(selecao.questoes_referencia, candidates, TipoDocumento.QUESTAO) if incluir_questoes else []
            ),
        )

    def _montar(self, ids: list[str], candidates: dict[str, SearchHit], tipo: TipoDocumento) -> list[TrechoRecuperado]:
        do_tipo = {i: h for i, h in candidates.items() if h.document.tipo == tipo}
        escolhidos = [i for i in dict.fromkeys(ids) if i in do_tipo]  # descarta ids inventados

        if len(escolhidos) < self._k:  # completa com os melhores scores não escolhidos
            restantes = sorted((i for i in do_tipo if i not in escolhidos), key=lambda i: -do_tipo[i].score)
            escolhidos += restantes[: self._k - len(escolhidos)]

        return [
            TrechoRecuperado(
                doc_id=i,
                tipo=tipo,
                fonte=do_tipo[i].document.fonte,
                texto=do_tipo[i].document.texto,
                score=do_tipo[i].score,
            )
            for i in escolhidos[: self._k]
        ]
