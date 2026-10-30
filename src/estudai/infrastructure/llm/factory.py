from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from estudai.settings import ModelConfig


def build_chat_model(
    config: ModelConfig, *, temperature: float, max_retries: int, timeout_s: float, api_key: str
) -> BaseChatModel:
    """Builds a compatible langchain chat model. Currently, only OpenAI formats are supported"""

    return ChatOpenAI(
        model=config.model,
        base_url=config.base_url,
        api_key=api_key,
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout_s,
    )
