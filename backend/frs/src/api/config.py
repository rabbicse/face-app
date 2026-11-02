import os

# vector database config
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
# Use ":memory:" for an embedded, non persistent Qdrant (tests / demos) or a path for local on-disk storage.
QDRANT_LOCATION = os.getenv('QDRANT_LOCATION')
# Full URL of a remote Qdrant, e.g. a Qdrant Cloud cluster (https://xyz.cloud.qdrant.io:6333) + its API key.
# Takes precedence over QDRANT_HOST / QDRANT_PORT.
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

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
