"""Configuração via variáveis de ambiente / `.env` (prefixo ESTUDAI_, aninhamento com `__`)."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    provider: Literal["gemini", "openai_compat"] = "gemini"
    model: str
    base_url: str | None = None
    json_mode: bool = True
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ESTUDAI_", env_nested_delimiter="__", env_file=".env", extra="ignore")

    # Níveis de capacidade (Seção 3)
    economic: ModelConfig = ModelConfig(model="gemini-3-flash")
    frontier: ModelConfig = ModelConfig(provider="openai_compat", model="muse-spark-1.3")
    # Painel de resolutores: idealmente modelos distintos; None = 3x o modelo de fronteira
    panel: list[ModelConfig] | None = None
    # Avaliação: juízes (preferencialmente outro fornecedor) e estudante simulado
    judges: list[ModelConfig] = [ModelConfig(model="gemini-3-flash")]
    student: ModelConfig | None = None  # None = modelo econômico

    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    embedding_provider: Literal["gemini", "hashing"] = "gemini"
    embedding_model: str = "gemini-embedding-001"

    data_dir: Path = Path("data")
    llm_timeout_s: float = 180.0
    llm_max_attempts: int = 4
    sandbox_timeout_s: float = 10.0

    # Limites de execução (Tabela 2 / Seção 4)
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
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def profiles_dir(self) -> Path:
        return self.data_dir / "profiles"
