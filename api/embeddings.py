from langchain_openai import OpenAIEmbeddings

from .config import get_settings

_embeddings = None


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        settings = get_settings()
        _embeddings = OpenAIEmbeddings(
            model=settings.EMBED_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )
    return _embeddings
