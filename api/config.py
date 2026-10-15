import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://app_user:app_password@db:5432/appdb",
    )
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    GEN_MODEL: str = os.getenv("GEN_MODEL", "gpt-4o-mini")
    CORPUS_VERSION: int = int(os.getenv("CORPUS_VERSION", "1"))


_settings = Settings()


def get_settings() -> Settings:
    return _settings


def validate_generation_settings(settings: Settings) -> None:
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required for answer generation.")
