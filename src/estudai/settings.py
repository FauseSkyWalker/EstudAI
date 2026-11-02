"""Configuração via variáveis de ambiente / `.env` (prefixo ESTUDAI_, aninhamento com `__`)."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    provider: Literal["gemini", "openai_compat"] = "gemini"
    model: str
    base_url: str | None = None
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0
    # Limite do fornecedor (ex.: plano gratuito do Gemini 3 Flash = 5/min). None = sem limitador.
    requests_per_minute: float | None = None


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

    embedding_provider: Literal["gemini", "sentencetransformers"] = "gemini"
    embedding_model: str = "gemini-embedding-001"
    embedding_query_params: dict[str, Any] = {}
    embedding_doc_params: dict[str, Any] = {}

    data_dir: Path = Path("data")
    llm_timeout_s: float = 180.0
    llm_max_attempts: int = 4
    sandbox_timeout_s: float = 10.0

    # Limites de execução
    kb_k: int = 4
    kb_min_score: float = 0.2
    kb_alpha: float = 0.5
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
    def embeddings_cache_path(self) -> Path:
        return self.data_dir / "processed" / "embeddings_cache"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def profiles_dir(self) -> Path:
        return self.data_dir / "profiles"

    @property
    def knowledge_db_uri(self) -> Path:
        return self.data_dir / "knowledge_db.lancedb"
