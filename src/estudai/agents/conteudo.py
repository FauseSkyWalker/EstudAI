"""Agente de Conteúdo (modelo econômico, ReAct): recupera e ranqueia material de apoio.

Validação de fidelidade: o agente apenas *seleciona* ids de documentos que as ferramentas
de fato retornaram; o texto entregue é sempre o original da base. Ids inventados são
descartados, então a saída é fiel às fontes por construção.
"""

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from estudai.agents.react import ReActRunner, Tool
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


class _BuscaArgs(BaseModel):
    consulta: str = Field(min_length=2)
    tipo: TipoDocumento | None = Field(default=None, description="'material', 'questao' ou null")


class SelecaoConteudo(BaseModel):
    trechos: list[str] = Field(default_factory=list, description="ids de material, em ordem de relevância")
    questoes_referencia: list[str] = Field(default_factory=list, description="ids de questões")
    justificativa: str = ""


class AgenteConteudo:
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
    ) -> None:
        self._kb = kb
        self._k = k
        self._min_score = min_score
        self._runner = ReActRunner(model, caller=self.papel.value, usage_cb=usage_cb, max_steps=max_steps)

    def buscar(self, pedido: str, topico: Topico | None = None, incluir_questoes: bool = True) -> PacoteMaterial:
        vistos: dict[str, SearchHit] = {}
        tools = [
            self._ferramenta(SearchMode.HIBRIDA, topico, vistos),
            self._ferramenta(SearchMode.SEMANTICA, topico, vistos),
            self._ferramenta(SearchMode.TEXTUAL, topico, vistos),
        ]
        tarefa = (
            f"Pedido: {pedido}\nTópico: {topico.value if topico else 'qualquer'}\n"
            f"Selecione até {self._k} trechos de material"
            + (f" e até {self._k} questões de referência." if incluir_questoes else ".")
        )
        try:
            selecao, _ = self._runner.run(SYSTEM, tarefa, tools, SelecaoConteudo)
        except Exception:  # noqa: BLE001 - fallback: ranking puro da base
            selecao = SelecaoConteudo()

        if not vistos:  # o modelo não chegou a buscar: consulta direta
            for hit in self._kb.search(pedido, SearchMode.HIBRIDA, k=2 * self._k, topico=topico):
                if hit.score >= self._min_score:
                    vistos[hit.document.id] = hit

        return PacoteMaterial(
            pedido=pedido,
            trechos=self._montar(selecao.trechos, vistos, TipoDocumento.MATERIAL),
            questoes_referencia=(
                self._montar(selecao.questoes_referencia, vistos, TipoDocumento.QUESTAO) if incluir_questoes else []
            ),
        )

    def _ferramenta(self, modo: SearchMode, topico: Topico | None, vistos: dict[str, SearchHit]) -> Tool:
        def buscar(args: _BuscaArgs) -> str:
            hits = self._kb.search(args.consulta, modo, k=2 * self._k, tipo=args.tipo, topico=topico)
            hits = [h for h in hits if h.score >= self._min_score]
            if not hits:
                return f"Nenhum resultado com score >= {self._min_score}."
            linhas = []
            for h in hits:
                vistos.setdefault(h.document.id, h)
                trecho = h.document.texto[:350].replace("\n", " ")
                linhas.append(f"[{h.document.id}] tipo={h.document.tipo.value} score={h.score:.2f} :: {trecho}")
            return "\n".join(linhas)

        return Tool(
            name=f"busca_{modo.value}",
            description=f"Busca {modo.value} na base de conhecimento (material didático e questões).",
            args_model=_BuscaArgs,
            fn=buscar,
        )

    def _montar(self, ids: list[str], vistos: dict[str, SearchHit], tipo: TipoDocumento) -> list[TrechoRecuperado]:
        do_tipo = {i: h for i, h in vistos.items() if h.document.tipo == tipo}
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
