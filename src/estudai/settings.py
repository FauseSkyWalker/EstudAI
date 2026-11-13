"""Configuração via variáveis de ambiente / `.env` (prefixo ESTUDAI_, aninhamento com `__`)."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    provider: Literal["gemini", "openai_compat", "openrouter", "claude_cli"] = "gemini"
    model: str
    base_url: str | None = None
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0
    # Limite do fornecedor (ex.: plano gratuito do Gemini 3 Flash = 5/min). None = sem limitador.
    requests_per_minute: float | None = None
    # Esforço de raciocínio (só claude_cli): low | medium | high | xhigh | max. None = padrão do modelo.
    effort: str | None = None
    # Como pedir JSON ao modelo (provedores compatíveis com OpenAI). Nem todo modelo do OpenRouter ou
    # endpoint próprio aceita json_schema; nesses, use function_calling ou json_mode.
    structured_output_method: Literal["json_schema", "function_calling", "json_mode"] = "json_schema"
    # Corpo extra enviado ao endpoint (ex.: {"thinking": {"type": "disabled"}} no DeepSeek).
    extra_body: dict[str, Any] = Field(default_factory=dict)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ESTUDAI_", env_nested_delimiter="__", env_file=".env", extra="ignore")

    economic: ModelConfig = ModelConfig(model="gemini-3-flash-preview")
    frontier: ModelConfig = ModelConfig(provider="openai_compat", model="muse-spark-1.3")
    # Painel de resolutores
    panel: list[ModelConfig] | None = None
    # Avaliação - juízes e simulador do aluno
    judges: list[ModelConfig] = [ModelConfig(model="gemini-3-flash-preview")]
    student: ModelConfig | None = None  # None = modelo econômico

    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None

    embedding_provider: Literal["gemini", "sentencetransformers"] = "gemini"
    embedding_model: str = "gemini-embedding-001"
    embedding_query_params: dict[str, Any] = {}
    embedding_doc_params: dict[str, Any] = {}

    data_dir: Path = Path("data")
    llm_timeout_s: float = 180.0
    llm_max_attempts: int = 4
    sandbox_timeout_s: float = 10.0
    figuras: bool = True
    figuras_timeout_s: float = 30.0
    max_figuras_tutor: int = 2

    # Limites de execução
    kb_k: int = 4
    react_max_steps: int = 6
    max_perguntas_diagnostico: int = 6
    max_rodadas_tutoria: int = 6
    exercicios_por_sessao: int = 8
    tamanho_painel: int = 3

    @property
    def questoes_path(self) -> Path:
        return self.data_dir / "processed" / "questoes.jsonl"

    @property
    def trechos_path(self) -> Path:
        return self.data_dir / "processed" / "trechos.jsonl"

    @property
    def rotulos_cache_path(self) -> Path:
        return self.data_dir / "processed" / "rotulos_cache.jsonl"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def profiles_dir(self) -> Path:
        return self.data_dir / "profiles"

    @property
    def knowledge_db_uri(self) -> Path:
        return self.data_dir / "knowledge_db.lancedb"
