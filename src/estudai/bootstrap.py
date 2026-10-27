"""Raiz de composição: o ÚNICO módulo que conhece as implementações concretas.

Monta adaptadores (LLMs, RAG, persistência, sandbox), injeta-os nos agentes e entrega um
`Orquestrador` pronto. Trocar fornecedor, banco vetorial ou executor é mudar só este arquivo.
"""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal

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
from estudai.infrastructure.knowledge.embedders import HashingEmbedder
from estudai.infrastructure.knowledge.hybrid import HybridKnowledgeBase
from estudai.infrastructure.knowledge.question_bank import JsonlQuestionBank
from estudai.infrastructure.llm.decorators import RetryingLLMClient, TrackingLLMClient
from estudai.infrastructure.persistence.json_profiles import JsonProfileRepository
from estudai.infrastructure.sandbox.python_executor import SubprocessPythonExecutor
from estudai.ports.knowledge import KnowledgeBase, QuestionBank
from estudai.ports.llm import Embedder, LLMClient
from estudai.ports.repositories import ProfileRepository
from estudai.ports.tools import CodeExecutor
from estudai.settings import ModelConfig, Settings

FonteExercicios = Literal["gerado", "banco"]


def _segredo(valor) -> str:
    if valor is None:
        raise RuntimeError("credencial ausente: defina a chave correspondente no .env")
    return valor.get_secret_value()


@dataclass
class Container:
    settings: Settings
    ledger: UsageLedger = field(default_factory=UsageLedger)

    # --- LLMs ---------------------------------------------------------------------------

    def llm(self, config: ModelConfig, ledger: UsageLedger | None = None) -> LLMClient:
        """`ledger` separado para chamadas fora do sistema (aluno simulado, juízes),
        para que não contaminem as métricas de custo/latência do pipeline."""
        s = self.settings
        if config.provider == "gemini":
            from estudai.infrastructure.llm.gemini import GeminiClient

            base: LLMClient = GeminiClient(config.model, _segredo(s.gemini_api_key), timeout_s=s.llm_timeout_s)
        else:
            from estudai.infrastructure.llm.openai_compat import OpenAICompatClient

            base = OpenAICompatClient(
                config.model,
                _segredo(s.openai_api_key),
                base_url=config.base_url,
                timeout_s=s.llm_timeout_s,
                json_mode=config.json_mode,
            )
        return TrackingLLMClient(
            RetryingLLMClient(base, max_attempts=s.llm_max_attempts),
            ledger or self.ledger,
            price_input_per_mtok=config.price_input_per_mtok,
            price_output_per_mtok=config.price_output_per_mtok,
        )

    @cached_property
    def llm_fronteira(self) -> LLMClient:
        return self.llm(self.settings.frontier)

    @cached_property
    def llm_economico(self) -> LLMClient:
        return self.llm(self.settings.economic)

    # --- dados e ferramentas -------------------------------------------------------------

    @cached_property
    def embedder(self) -> Embedder:
        if self.settings.embedding_provider == "hashing":
            return HashingEmbedder()
        from estudai.infrastructure.llm.gemini import GeminiEmbedder

        return GeminiEmbedder(self.settings.embedding_model, _segredo(self.settings.gemini_api_key))

    @cached_property
    def kb(self) -> KnowledgeBase:
        return HybridKnowledgeBase.load(self.settings.index_dir, self.embedder, alpha=self.settings.kb_alpha)

    @cached_property
    def banco(self) -> QuestionBank:
        return JsonlQuestionBank.load(self.settings.questoes_path)

    @cached_property
    def executor(self) -> CodeExecutor:
        return SubprocessPythonExecutor(timeout_s=self.settings.sandbox_timeout_s)

    def perfis(self, raiz: Path | None = None) -> ProfileRepository:
        return JsonProfileRepository(raiz or self.settings.profiles_dir)

    # --- agentes -------------------------------------------------------------------------

    @cached_property
    def conteudo(self) -> AgenteConteudo:
        s = self.settings
        return AgenteConteudo(self.llm_economico, self.kb, k=s.kb_k, min_score=s.kb_min_score)

    @cached_property
    def validador(self) -> ValidadorExercicios:
        s = self.settings
        configs = s.panel or [s.frontier] * s.tamanho_painel
        # temperatura > 0 para que resolutores do mesmo modelo sejam amostras independentes
        resolutores = [Resolutor(self.llm(c), temperature=0.7) for c in configs]
        return ValidadorExercicios(PainelResolutores(resolutores), VerificadorSimbolico(self.executor))

    def fornecedor(self, fonte: FonteExercicios, seed: int = 0) -> ExerciseProvider:
        if fonte == "banco":
            return BankExerciseProvider(self.banco, seed=seed)
        gerador = GeradorExercicios(
            self.llm_fronteira, self.conteudo, self.executor, max_steps=self.settings.react_max_steps
        )
        return GeneratedExerciseProvider(gerador, self.validador)

    def orquestrador(
        self,
        config: ConfigSessao | None = None,
        fonte: FonteExercicios = "gerado",
        perfis: ProfileRepository | None = None,
    ) -> Orquestrador:
        s = self.settings
        config = config or ConfigSessao(
            max_rodadas_tutoria=s.max_rodadas_tutoria, exercicios_por_sessao=s.exercicios_por_sessao
        )
        return Orquestrador(
            diagnosticador=Diagnosticador(self.llm_fronteira, max_perguntas=s.max_perguntas_diagnostico),
            tutores={d: Tutor(self.llm_fronteira, d) for d in Disciplina},
            conteudo=self.conteudo,
            resumo=AgenteResumo(self.llm_economico),
            corretor=Corretor(self.llm_fronteira),
            fornecedor=self.fornecedor(fonte, seed=config.seed),
            validador=self.validador,
            perfis=perfis or self.perfis(),
            banco=self.banco,
            config=config,
            ledger=self.ledger,
        )
