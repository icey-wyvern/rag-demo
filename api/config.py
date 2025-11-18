import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
	DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+psycopg://app_user:app_password@db:5432/appdb")
	OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
	EMBED_MODEL: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
	GEN_MODEL: str = os.getenv("GEN_MODEL", "gpt-4o-mini")

_settings = Settings()

def get_settings() -> Settings:
	return _settings
