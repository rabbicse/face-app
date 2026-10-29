import base64
from typing import Iterable, Sequence

import numpy as np

EMBEDDING_DIM = 512


def l2_normalize(vector: Sequence[float], eps: float = 1e-10) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm < eps:
        raise ValueError("embedding has zero or non-finite norm")
    return v / norm


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def fuse_embeddings(vectors: Iterable[Sequence[float]]) -> np.ndarray:
    """Mean of L2 normalised embeddings, re-normalised (a person 'template')."""
    stacked = np.stack([l2_normalize(v) for v in vectors])
    return l2_normalize(stacked.mean(axis=0))


def encode_base64(vector: Sequence[float]) -> str:
    """Little endian float32 -> base64 (same wire format as the web and mobile clients)."""
    return base64.b64encode(np.asarray(vector, dtype='<f4').tobytes()).decode('ascii')


def decode_base64(data: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(data), dtype='<f4').astype(np.float32)
