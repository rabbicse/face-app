from functools import lru_cache

from api import config


def get_face_service():
    # imported lazily: pulls in torch and the model weights (only needed for server side inference)
    from api.services.face_service import FaceService

    return FaceService(det_model_path=config.DETECTION_MODEL_PATH,
                       det_model_tar=config.DETECTION_MODEL_TAR_PATH,
                       det_network=config.DETECTION_NETWORK,
                       rec_model_path=config.RECOGNITION_MODEL_PATH,
                       rec_network=config.RECOGNITION_NETWORK,
                       device=config.DEVICE)


def get_cache_service():
    from cache.redis_service import RedisCacheService

    return RedisCacheService()


def get_vector_db_context():
    from vector_db.qdrant_context import VectorDbContext

    return VectorDbContext(host=config.QDRANT_HOST,
                           port=config.QDRANT_PORT,
                           collection_name=config.QDRANT_COLLECTION_NAME)


@lru_cache(maxsize=1)
def get_qdrant_client():
    from qdrant_client import QdrantClient

    if config.QDRANT_LOCATION:
        return QdrantClient(location=config.QDRANT_LOCATION) if config.QDRANT_LOCATION == ':memory:' \
            else QdrantClient(path=config.QDRANT_LOCATION)
    return QdrantClient(url=f'http://{config.QDRANT_HOST}:{config.QDRANT_PORT}')


@lru_cache(maxsize=1)
def get_embedding_service():
    from api.services.embedding_service import EmbeddingService
    from vector_db.embedding_store import EmbeddingStore

    store = EmbeddingStore(get_qdrant_client(), collection_prefix=config.EMBEDDING_COLLECTION_PREFIX,
                           dim=config.EMBEDDING_DIM)
    return EmbeddingService(store, dim=config.EMBEDDING_DIM, models=config.EMBEDDING_MODELS,
                            threshold=config.MATCH_THRESHOLD, margin=config.MATCH_MARGIN)
