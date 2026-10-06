from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str | None = None
    llm_model: str = "gpt-4.1-mini"
    search_provider: str = "demo"
    tavily_api_key: str | None = None
    serper_api_key: str | None = None
    brave_api_key: str | None = None
    backend_port: int = 8000
    frontend_port: int = 5173
    max_research_rounds: int = 2
    max_results_per_query: int = 3
    max_sources: int = 12

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parents[2] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def demo_mode(self) -> bool:
        return not self.openai_api_key or self.search_provider.lower() == "demo"


@lru_cache
def get_settings() -> Settings:
    return Settings()

