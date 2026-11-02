from functools import lru_cache

from api import config


@lru_cache(maxsize=1)
def get_qdrant_client():
    from qdrant_client import QdrantClient

    if config.QDRANT_LOCATION:
        return QdrantClient(location=config.QDRANT_LOCATION) if config.QDRANT_LOCATION == ':memory:' \
            else QdrantClient(path=config.QDRANT_LOCATION)
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None)
    return QdrantClient(url=f'http://{config.QDRANT_HOST}:{config.QDRANT_PORT}')


@lru_cache(maxsize=1)
def get_embedding_service():
    from api.services.embedding_service import EmbeddingService
    from vector_db.embedding_store import EmbeddingStore

    store = EmbeddingStore(get_qdrant_client(), collection_prefix=config.EMBEDDING_COLLECTION_PREFIX,
                           dim=config.EMBEDDING_DIM)
    return EmbeddingService(store, dim=config.EMBEDDING_DIM, models=config.EMBEDDING_MODELS,
                            threshold=config.MATCH_THRESHOLD, margin=config.MATCH_MARGIN)
