"""Raiz de composição: o ÚNICO módulo que conhece as implementações concretas.

Monta adaptadores (LLMs, RAG, persistência, sandbox), injeta-os nos agentes e entrega um
`Orquestrador` pronto. Trocar fornecedor, banco vetorial ou executor é mudar só este arquivo.
"""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from estudai.agents.conteudo import AgenteConteudo
from estudai.agents.corretor import Corretor
from estudai.agents.diagnosticador import Diagnosticador
from estudai.agents.gerador import GeradorExercicios
from estudai.agents.resolutor import Resolutor
from estudai.agents.resumo import AgenteResumo
from estudai.agents.tutor import Tutor
from estudai.application.orquestrador import ConfigSessao, Orquestrador
from estudai.application.pratica import BankExerciseProvider, ExerciseProvider, GeneratedExerciseProvider
from estudai.application.telemetria import UsageLedger
from estudai.application.validacao import PainelResolutores, ValidadorExercicios, VerificadorSimbolico
from estudai.domain.curriculum import Disciplina
from estudai.domain.figura import TipoFigura
from estudai.domain.mensagens import Papel
from estudai.infrastructure.figures import CompositeFigureRenderer, MatplotlibRenderer, MermaidRenderer
from estudai.infrastructure.knowledge.database import KnowledgeDatabase
from estudai.infrastructure.knowledge.embedders import SentenceTransformersEmbedder
from estudai.infrastructure.knowledge.question_bank import JsonlQuestionBank
from estudai.infrastructure.llm.callbacks import UsageLedgerCallback
from estudai.infrastructure.llm.factory import build_chat_model
from estudai.infrastructure.persistence.json_profiles import JsonProfileRepository
from estudai.infrastructure.sandbox.python_executor import SubprocessPythonExecutor
from estudai.ports.figures import FigureRenderer
from estudai.ports.knowledge import KnowledgeBase, QuestionBank
from estudai.ports.llm import Embedder
from estudai.ports.repositories import ProfileRepository
from estudai.ports.tools import CodeExecutor
from estudai.settings import ModelConfig, Settings

FonteExercicios = Literal["gerado", "banco"]

# Temperatura é decisão de composição, por papel:
# - resolutores > 0 para que três instâncias do mesmo modelo sejam amostras independentes
#   (senão os votos ficam correlacionados e o consenso do painel perde valor);
# - juízes e classificador em 0 para avaliações reprodutíveis.
TEMPERATURA: dict[Papel, float] = {
    Papel.DIAGNOSTICADOR: 0.2,
    Papel.TUTOR: 0.4,
    Papel.CONTEUDO: 0.3,
    Papel.RESUMO: 0.2,
    Papel.GERADOR_EXERCICIOS: 0.7,
    Papel.CORRETOR: 0.2,
    Papel.PAINEL_RESOLUTORES: 0.7,
    Papel.CLASSIFICADOR: 0.0,
    Papel.ALUNO_SIMULADO: 0.7,
    Papel.JUIZ: 0.0,
}


def _segredo(valor) -> str:
    if valor is None:
        raise RuntimeError("credencial ausente: defina a chave correspondente no .env")
    return valor.get_secret_value()


@dataclass
class Container:
    settings: Settings
    ledger: UsageLedger = field(default_factory=UsageLedger)
    _modelos: dict = field(default_factory=dict, repr=False)
    _limitadores: dict = field(default_factory=dict, repr=False)

    # --- LLMs ---------------------------------------------------------------------------

    def chat(self, config: ModelConfig, papel: Papel, ledger: UsageLedger | None = None) -> BaseChatModel:
        """Modelo de chat para um papel. `ledger` separado serve para chamadas fora do sistema
        (aluno simulado, juízes), para que não contaminem as métricas de custo/latência."""
        ledger = ledger or self.ledger
        chave = (config.model_dump_json(), TEMPERATURA[papel], id(ledger))
        if chave not in self._modelos:
            s = self.settings
            chave_api = {
                "gemini": s.gemini_api_key,
                "openai_compat": s.openai_api_key,
                "openrouter": s.openrouter_api_key,
            }.get(config.provider)
            self._modelos[chave] = build_chat_model(
                provider=config.provider,
                model=config.model,
                base_url=config.base_url,
                api_key=None if config.provider == "claude_cli" else _segredo(chave_api),
                effort=config.effort,
                structured_output_method=config.structured_output_method,
                extra_body=config.extra_body,
                temperature=TEMPERATURA[papel],
                max_retries=s.llm_max_attempts - 1,
                timeout_s=s.llm_timeout_s,
                rate_limiter=self._limitador(config),
                callbacks=[
                    UsageLedgerCallback(
                        ledger,
                        model=config.model,
                        price_in=config.price_input_per_mtok,
                        price_out=config.price_output_per_mtok,
                    )
                ],
            )
        return self._modelos[chave]

    def _limitador(self, config: ModelConfig) -> InMemoryRateLimiter | None:
        """Um limitador por modelo, compartilhado entre papéis, temperaturas e ledgers: a cota
        do fornecedor é por modelo, então juízes, aluno simulado e agentes disputam a mesma."""
        if not config.requests_per_minute:
            return None
        chave = (config.provider, config.model, config.base_url)
        if chave not in self._limitadores:
            self._limitadores[chave] = InMemoryRateLimiter(
                requests_per_second=config.requests_per_minute / 60,
                check_every_n_seconds=0.2,
                max_bucket_size=1,  # sem rajadas: espaça as chamadas uniformemente
            )
        return self._limitadores[chave]

    def fronteira(self, papel: Papel) -> BaseChatModel:
        return self.chat(self.settings.frontier, papel)

    def economico(self, papel: Papel) -> BaseChatModel:
        return self.chat(self.settings.economic, papel)

    def classificador(self) -> BaseChatModel:
        """function calling é problemático com saída em LaTeX, logo usamos json_mode
        para classificação/normalização da base inicial
        """
        config = self.settings.economic.model_copy(update={"structured_output_method": "json_mode"})
        return self.chat(config, Papel.CLASSIFICADOR)

    # --- dados e ferramentas -------------------------------------------------------------

    @cached_property
    def embedder(self) -> Embedder:
        if self.settings.embedding_provider == "sentencetransformers":
            return SentenceTransformersEmbedder(
                self.settings.embedding_model, self.settings.embedding_query_params, self.settings.embedding_doc_params
            )
        from estudai.infrastructure.llm.gemini import GeminiEmbedder

        return GeminiEmbedder(self.settings.embedding_model, _segredo(self.settings.gemini_api_key))

    @cached_property
    def kb(self) -> KnowledgeBase:
        return KnowledgeDatabase(self.settings.knowledge_db_uri, self.embedder)

    @cached_property
    def banco(self) -> QuestionBank:
        return JsonlQuestionBank.load(self.settings.questoes_path)

    @cached_property
    def executor(self) -> CodeExecutor:
        return SubprocessPythonExecutor(timeout_s=self.settings.sandbox_timeout_s)

    @cached_property
    def figuras(self) -> FigureRenderer | None:
        """Renderiza figuras do Tutor (matplotlib na sandbox e Mermaid). None se desligado."""
        if not self.settings.figuras:
            return None
        return CompositeFigureRenderer(
            {
                TipoFigura.GRAFICO: MatplotlibRenderer(timeout_s=self.settings.figuras_timeout_s),
                TipoFigura.DIAGRAMA: MermaidRenderer(),
            }
        )

    def dados_faltando(self) -> list[tuple[Path, str]]:
        """(arquivo ausente, comando que o gera) — para as interfaces falharem cedo, antes de gastar LLM."""
        s = self.settings
        necessarios = [
            (s.questoes_path, "uv run estudai ingest-questions"),
            (s.knowledge_db_uri, "uv run estudai build-index"),
        ]
        return [(caminho, comando) for caminho, comando in necessarios if not caminho.exists()]

    def perfis(self, raiz: Path | None = None) -> ProfileRepository:
        return JsonProfileRepository(raiz or self.settings.profiles_dir)

    # --- agentes -------------------------------------------------------------------------

    @cached_property
    def conteudo(self) -> AgenteConteudo:
        s = self.settings
        return AgenteConteudo(self.economico(Papel.CONTEUDO), self.kb, k=s.kb_k)

    @cached_property
    def validador(self) -> ValidadorExercicios:
        s = self.settings
        configs = s.panel or [s.frontier] * s.tamanho_painel
        resolutores = [Resolutor(self.chat(c, Papel.PAINEL_RESOLUTORES)) for c in configs]
        return ValidadorExercicios(PainelResolutores(resolutores), VerificadorSimbolico(self.executor))

    def fornecedor(self, fonte: FonteExercicios, seed: int = 0) -> ExerciseProvider:
        if fonte == "banco":
            return BankExerciseProvider(self.banco, seed=seed)
        gerador = GeradorExercicios(
            self.fronteira(Papel.GERADOR_EXERCICIOS),
            self.conteudo,
            self.executor,
            max_steps=self.settings.react_max_steps,
        )
        return GeneratedExerciseProvider(gerador, self.validador)

    def orquestrador(
        self,
        config: ConfigSessao | None = None,
        fonte: FonteExercicios = "gerado",
        perfis: ProfileRepository | None = None,
        com_figuras: bool | None = None,
    ) -> Orquestrador:
        s = self.settings
        config = config or ConfigSessao(
            max_rodadas_tutoria=s.max_rodadas_tutoria, exercicios_por_sessao=s.exercicios_por_sessao
        )
        # `com_figuras=False` desliga tanto o renderer quanto a instrução no prompt do Tutor
        usar_figuras = s.figuras if com_figuras is None else com_figuras
        return Orquestrador(
            diagnosticador=Diagnosticador(
                self.fronteira(Papel.DIAGNOSTICADOR), max_perguntas=s.max_perguntas_diagnostico
            ),
            tutores={
                d: Tutor(
                    self.fronteira(Papel.TUTOR),
                    d,
                    figuras=usar_figuras,
                    max_figuras=s.max_figuras_tutor,
                )
                for d in Disciplina
            },
            conteudo=self.conteudo,
            resumo=AgenteResumo(self.economico(Papel.RESUMO)),
            corretor=Corretor(self.fronteira(Papel.CORRETOR)),
            fornecedor=self.fornecedor(fonte, seed=config.seed),
            validador=self.validador,
            perfis=perfis or self.perfis(),
            banco=self.banco,
            config=config,
            ledger=self.ledger,
            figuras=self.figuras if usar_figuras else None,
            max_figuras=s.max_figuras_tutor,
        )
