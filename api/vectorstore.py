from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

from .config import get_settings

_settings = get_settings()

# Shared embeddings instance
_embeddings = OpenAIEmbeddings(
	model=_settings.EMBED_MODEL,
	openai_api_key=_settings.OPENAI_API_KEY,
)

# Singleton PGVector store (uses the same Postgres as the rest of the app)
_vectorstore = PGVector(
	embeddings=_embeddings,
	collection_name="govuk",
	connection=_settings.DATABASE_URL,  # e.g. postgresql+psycopg://app_user:app_password@db:5432/appdb
	use_jsonb=True,
)

def get_vectorstore() -> PGVector:
	return _vectorstore
