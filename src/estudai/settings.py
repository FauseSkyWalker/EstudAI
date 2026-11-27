"""Configuração via variáveis de ambiente / `.env` (prefixo ESTUDAI_, aninhamento com `__`)."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    provider: Literal["gemini", "openai_compat", "openrouter"] = "gemini"
    model: str
    base_url: str | None = None
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0
    requests_per_minute: float | None = None
    structured_output_method: Literal["json_schema", "function_calling", "json_mode"] = "json_schema"
    extra_body: dict[str, Any] = Field(default_factory=dict)
    default_headers: dict[str, str] = Field(default_factory=dict)  # required for opencode go stuff


def _deepseek_flash() -> ModelConfig:
    return ModelConfig(
        provider="openai_compat",
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        price_input_per_mtok=0.30,
        price_output_per_mtok=2.50,
        structured_output_method="function_calling",
        extra_body={"thinking": {"type": "disabled"}},
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ESTUDAI_", env_nested_delimiter="__", env_file=".env", extra="ignore")

    economic: ModelConfig = Field(default_factory=_deepseek_flash)
    frontier: ModelConfig = Field(default_factory=_deepseek_flash)
    # Painel de resolutores
    panel: list[ModelConfig] | None = None
    # Avaliação - juízes e simulador do aluno
    judges: list[ModelConfig] = Field(default_factory=lambda: [ModelConfig(provider="gemini", model="gemini-3-flash")])
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
    max_figuras_exercicio: int = 1

    # Limites de execução
    kb_k: int = 4
    react_max_steps: int = 8
    max_perguntas_diagnostico: int = 6
    max_rodadas_tutoria: int = 6
    tutor_max_steps: int = 3
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
    def profiles_dir(self) -> Path:
        return self.data_dir / "profiles"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def knowledge_db_uri(self) -> Path:
        return self.data_dir / "knowledge_db.lancedb"
