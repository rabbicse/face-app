import os


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ('1', 'true', 'yes', 'on')


# ---- legacy server side inference (RetinaFace + ArcFace r100 on PyTorch) ----
# Set FRS_ENABLE_SERVER_INFERENCE=false to run a lightweight API (no torch / model weights
# needed) that only stores and searches embeddings produced on the client devices.
ENABLE_SERVER_INFERENCE = _env_bool('FRS_ENABLE_SERVER_INFERENCE', True)

# DETECTION_NETWORK = 'mobile0.25'
DETECTION_NETWORK = os.getenv('FRS_DETECTION_NETWORK', 'resnet50')

DETECTION_MODEL_PATH = os.path.abspath(os.getenv('FRS_DETECTION_MODEL_PATH', '../../models/Resnet50_Final.pth'))
# DETECTION_MODEL_PATH = os.path.abspath('../../models/mobilenet0.25_Final.pth')
DETECTION_MODEL_TAR_PATH = os.path.abspath(
    os.getenv('FRS_DETECTION_MODEL_TAR_PATH', '../../models/mobilenetV1X0.25_pretrain.tar'))

RECOGNITION_MODEL_PATH = os.path.abspath(
    os.getenv('FRS_RECOGNITION_MODEL_PATH', '../../models/backbone_glint360k_r100.pth'))
RECOGNITION_NETWORK = os.getenv('FRS_RECOGNITION_NETWORK', 'r100')
# DEVICE = 'cuda:0'
DEVICE = os.getenv('FRS_DEVICE', 'cpu')


# vector database config
# QDRANT_HOST = '192.168.97.67'
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'face_collection')
# Use ":memory:" for an embedded, non persistent Qdrant (tests / demos) or a path for local on-disk storage.
QDRANT_LOCATION = os.getenv('QDRANT_LOCATION')

# ---- client side embeddings ----
# Embeddings from different models live in different vector spaces, so every model gets its own
# collection: f'{EMBEDDING_COLLECTION_PREFIX}{model}'.
EMBEDDING_COLLECTION_PREFIX = os.getenv('FRS_EMBEDDING_COLLECTION_PREFIX', 'face_embeddings_')
EMBEDDING_DIM = int(os.getenv('FRS_EMBEDDING_DIM', 512))
# Models the API accepts embeddings from (comma separated). All clients must use the same one.
EMBEDDING_MODELS = [m.strip() for m in os.getenv('FRS_EMBEDDING_MODELS', 'w600k_mbf').split(',') if m.strip()]
# Cosine similarity needed to accept an identification (MobileFaceNet / w600k_mbf, aligned faces).
MATCH_THRESHOLD = float(os.getenv('FRS_MATCH_THRESHOLD', 0.45))
# Minimum gap between the best and the second best *person*; rejects ambiguous matches.
MATCH_MARGIN = float(os.getenv('FRS_MATCH_MARGIN', 0.05))

CORS_ORIGINS = [o.strip() for o in os.getenv('FRS_CORS_ORIGINS', '*').split(',') if o.strip()]
