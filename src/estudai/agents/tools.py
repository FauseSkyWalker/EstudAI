from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from estudai.domain.curriculum import Topico
from estudai.domain.material import TipoDocumento
from estudai.ports.knowledge import KnowledgeBase
from estudai.ports.tools import CodeExecutor

if TYPE_CHECKING:
    from estudai.agents.conteudo import AgenteConteudo


class _SearchArgs(BaseModel):
    consulta: str = Field(min_length=2)
    tipo: TipoDocumento | None = Field(default=None, description="'material', 'questao' ou null")


class _ConsultaArgs(BaseModel):
    pedido: str = Field(min_length=3, description="O que o agente de conteúdo deve buscar")


class _PythonArgs(BaseModel):
    code: str = Field(description="Código em formato de string")
    timeout_s: float = Field(default=30, gt=0, description="Timeout, em segundos")


def search_tool(modo, topico, candidatos, kb: KnowledgeBase, k, min_score) -> StructuredTool:
    def buscar(consulta: str, tipo: TipoDocumento | None = None) -> str:
        hits = kb.search(consulta, modo, k=2 * k, tipo=tipo, topico=topico)

        hits = [h for h in hits if h.score >= min_score]
        if not hits:
            return f"Nenhum resultado com score >= {min_score}."

        linhas = []
        for h in hits:
            candidatos.setdefault(h.document.id, h)
            trecho = h.document.texto[:350].replace("\n", " ")
            linhas.append(f"[{h.document.id}] tipo={h.document.tipo.value} score={h.score:.2f} :: {trecho}")

        return "\n".join(linhas)

    return StructuredTool.from_function(
        func=buscar,
        name=f"busca_{modo.value}",
        description=f"Busca {modo.value} na base de conhecimento.",
        args_schema=_SearchArgs,
    )


def call_conteudo_agent_tool(agente_conteudo: "AgenteConteudo", topico: Topico | None = None) -> StructuredTool:
    def consultar(pedido: str) -> str:
        return agente_conteudo.buscar(pedido, topico).como_contexto(max_chars=3000)

    return StructuredTool.from_function(
        func=consultar,
        name="consultar_conteudo",
        description="Pede ao Agente de Conteúdo material e questões de referência.",
        args_schema=_ConsultaArgs,
    )


def python_tool(executor: CodeExecutor) -> StructuredTool:
    def executar(code: str, timeout_s: float) -> str:
        result = executor.run(code, timeout_s)
        if result.timed_out:
            return "Tempo esgotado."
        return f"ok={result.ok}\nstdout:\n{result.stdout[-1500:]}\nstderr:\n{result.stderr[-800:]}"

    return StructuredTool.from_function(
        func=executar,
        name="executar_python",
        description="Executa Python isolado (sem rede, com timeout). Use print() para ver resultados.",
        args_schema=_PythonArgs,
    )
