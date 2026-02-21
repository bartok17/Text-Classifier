from app.services.embedding_service import EmbeddingClient, OllamaEmbeddingClient


def build_embedding_client() -> EmbeddingClient:
    return OllamaEmbeddingClient()
