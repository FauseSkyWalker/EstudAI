"""Agente Gerador de Exercícios (modelo de fronteira, ReAct).

Gera um item por vez para um conceito/dificuldade planejados. Ferramentas:
- `consultar_conteudo`: aciona o Agente de Conteúdo (questões de referência e material);
- `executar_python`: sandbox para conferir contas (sympy disponível).
O item ainda passa pelo painel de resolutores antes de ser entregue (application.validacao).
"""

from uuid import uuid4

from pydantic import BaseModel, Field

from estudai.agents.base import to_json
from estudai.agents.conteudo import AgenteConteudo
from estudai.agents.react import ReActRunner, Tool
from estudai.domain.curriculum import Topico
from estudai.domain.mensagens import Papel
from estudai.domain.planejamento import ItemPlanejado
from estudai.domain.questoes import Exercicio, OrigemItem
from estudai.ports.llm import LLMClient
from estudai.ports.tools import CodeExecutor

SYSTEM = """Você é o Agente Gerador de Exercícios de um sistema de tutoria para vestibulares
brasileiros. Crie UMA questão objetiva inédita no estilo ENEM/Fuvest/Unicamp.
Requisitos:
- Exatamente 5 alternativas, apenas UMA correta; distratores plausíveis, baseados em erros típicos.
- Sem pistas gramaticais ou de tamanho entre alternativas; enunciado claro e autocontido (sem figuras).
- Adeque ao conceito-alvo, à dificuldade e às lacunas do aluno.
- Confira as contas com `executar_python` antes de finalizar.
- Se a resposta for numérica, preencha `codigo_verificacao` com um script Python que faça
  `print()` APENAS do valor correto (número ou expressão sympy), para verificação automática.
- Use `consultar_conteudo` para se inspirar em questões reais e cite os ids em `referencias`.
- `alternativas` contém só o texto de cada opção, sem "A)" etc.
"""


class _ConsultaArgs(BaseModel):
    pedido: str = Field(min_length=3)


class _CodigoArgs(BaseModel):
    codigo: str = Field(min_length=1)


class RascunhoExercicio(BaseModel):
    enunciado: str
    alternativas: list[str] = Field(min_length=5, max_length=5)
    gabarito: str
    resolucao: str
    codigo_verificacao: str | None = None
    referencias: list[str] = Field(default_factory=list)


class GeradorExercicios:
    papel = Papel.GERADOR_EXERCICIOS

    def __init__(
        self,
        llm: LLMClient,
        conteudo: AgenteConteudo,
        executor: CodeExecutor,
        *,
        max_steps: int = 6,
    ) -> None:
        self._conteudo = conteudo
        self._executor = executor
        self._runner = ReActRunner(llm, caller=self.papel.value, max_steps=max_steps)

    def gerar(self, perfil: dict, topico: Topico, plano: ItemPlanejado, evitar: list[str]) -> Exercicio:
        tools = [self._ferramenta_conteudo(topico), self._ferramenta_python()]
        tarefa = (
            f"Tópico: {topico.value}\nConceito-alvo: {plano.conceito}\n"
            f"Dificuldade-alvo: {plano.dificuldade.value}\n"
            f"Perfil do aluno:\n{to_json(perfil)}\n"
            "Não repita estes enunciados já usados:\n" + ("\n".join(f"- {e[:160]}" for e in evitar) or "(nenhum)")
        )
        rascunho, _ = self._runner.run(SYSTEM, tarefa, tools, RascunhoExercicio, temperature=0.7)
        return Exercicio(
            id=f"gen-{uuid4().hex[:8]}",
            topico=topico,
            conceitos=[plano.conceito],
            dificuldade_alvo=plano.dificuldade,
            enunciado=rascunho.enunciado,
            alternativas=rascunho.alternativas,
            gabarito=rascunho.gabarito,
            resolucao=rascunho.resolucao,
            origem=OrigemItem.GERADO,
            referencias=rascunho.referencias,
            codigo_verificacao=rascunho.codigo_verificacao,
        )

    def _ferramenta_conteudo(self, topico: Topico) -> Tool:
        def consultar(args: _ConsultaArgs) -> str:
            return self._conteudo.buscar(args.pedido, topico).como_contexto(max_chars=3000)

        return Tool(
            name="consultar_conteudo",
            description="Pede ao Agente de Conteúdo material e questões de referência.",
            args_model=_ConsultaArgs,
            fn=consultar,
        )

    def _ferramenta_python(self) -> Tool:
        def executar(args: _CodigoArgs) -> str:
            r = self._executor.run(args.codigo)
            if r.timed_out:
                return "Tempo esgotado."
            return f"ok={r.ok}\nstdout:\n{r.stdout[-1500:]}\nstderr:\n{r.stderr[-800:]}"

        return Tool(
            name="executar_python",
            description="Executa Python isolado (sem rede, com timeout). Use print() para ver resultados.",
            args_model=_CodigoArgs,
            fn=executar,
        )
