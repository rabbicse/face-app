from api import config
from api.services.face_service import FaceService
from cache.cache_service import CacheService
from cache.redis_service import RedisCacheService
from vector_db.qdrant_context import VectorDbContext


def get_face_service() -> FaceService:
    return FaceService(det_model_path=config.DETECTION_MODEL_PATH,
                       det_model_tar=config.DETECTION_MODEL_TAR_PATH,
                       det_network=config.DETECTION_NETWORK,
                       rec_model_path=config.RECOGNITION_MODEL_PATH,
                       rec_network=config.RECOGNITION_NETWORK,
                       device=config.DEVICE)

def get_cache_service() -> RedisCacheService:
    return RedisCacheService()

def get_vector_db_context() -> VectorDbContext:
    return VectorDbContext()