from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from face_pipeline.embedding import decode_base64, fuse_embeddings, l2_normalize
from vector_db.embedding_store import EmbeddingStore, Hit, StoredVector


class EmbeddingValidationError(ValueError):
    pass


@dataclass
class PersonMatch:
    person_id: int
    score: float
    payload: Dict[str, Any]


@dataclass
class IdentifyResult:
    match: Optional[PersonMatch]
    candidates: List[PersonMatch]
    reason: str


def parse_embedding(value: Any, dim: int) -> np.ndarray:
    """Accept a list of floats or a base64 encoded little endian float32 buffer."""
    try:
        vec = decode_base64(value) if isinstance(value, str) else np.asarray(value, dtype=np.float32)
    except Exception as x:
        raise EmbeddingValidationError(f'invalid embedding encoding: {x}')
    vec = vec.reshape(-1)
    if vec.shape[0] != dim:
        raise EmbeddingValidationError(f'embedding must have {dim} dimensions, got {vec.shape[0]}')
    if not np.all(np.isfinite(vec)):
        raise EmbeddingValidationError('embedding contains NaN or infinite values')
    try:
        return l2_normalize(vec)
    except ValueError as x:
        raise EmbeddingValidationError(str(x))


def best_per_person(hits: Sequence[Hit]) -> List[PersonMatch]:
    """Collapse point hits to one entry per person (max score), sorted best first."""
    best: Dict[int, Hit] = {}
    for h in hits:
        if h.person_id not in best or h.score > best[h.person_id].score:
            best[h.person_id] = h
    ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)
    return [PersonMatch(person_id=h.person_id, score=h.score, payload=h.payload) for h in ranked]


def decide(candidates: Sequence[PersonMatch], threshold: float, margin: float) -> IdentifyResult:
    if not candidates:
        return IdentifyResult(match=None, candidates=[], reason='no_enrolled_faces')
    best = candidates[0]
    if best.score < threshold:
        return IdentifyResult(match=None, candidates=list(candidates), reason='below_threshold')
    if len(candidates) > 1 and best.score - candidates[1].score < margin:
        return IdentifyResult(match=None, candidates=list(candidates), reason='ambiguous')
    return IdentifyResult(match=best, candidates=list(candidates), reason='matched')


class EmbeddingService:
    def __init__(self, store: EmbeddingStore, dim: int, models: Sequence[str],
                 threshold: float, margin: float):
        self.store = store
        self.dim = dim
        self.models = list(models)
        self.threshold = threshold
        self.margin = margin

    def check_model(self, model: str) -> None:
        if model not in self.models:
            raise EmbeddingValidationError(f'unsupported embedding model "{model}", expected one of {self.models}')

    def register(self, model: str, person: Dict[str, Any], samples: Sequence[Dict[str, Any]],
                 min_consistency: float = 0.3) -> Dict[str, Any]:
        self.check_model(model)
        if not samples:
            raise EmbeddingValidationError('at least one sample is required')

        vectors = [parse_embedding(s['embedding'], self.dim) for s in samples]
        template = fuse_embeddings(vectors)

        # leave-one-out check: every sample must look like the same person as the other samples
        # (guards against a different face appearing during the guided capture)
        consistency = [1.0] if len(vectors) == 1 else \
            [float(np.dot(v, fuse_embeddings(vectors[:i] + vectors[i + 1:]))) for i, v in enumerate(vectors)]
        outliers = [i for i, c in enumerate(consistency) if c < min_consistency]
        if outliers:
            raise EmbeddingValidationError(f'samples {outliers} do not match the other samples '
                                           f'(cosine < {min_consistency})')

        stored = [StoredVector(vector=v, kind='sample', pose_label=s.get('pose_label'), pose=s.get('pose'),
                               quality=s.get('quality')) for v, s in zip(vectors, samples)]
        stored.append(StoredVector(vector=template, kind='template'))
        count = self.store.replace_person(model, person, stored)
        return {'stored_vectors': count, 'min_consistency': round(min(consistency), 4)}

    def identify(self, model: str, embedding: Any, top_k: int = 5,
                 threshold: Optional[float] = None) -> IdentifyResult:
        self.check_model(model)
        vec = parse_embedding(embedding, self.dim)
        # each person owns several points, so fetch enough to rank top_k distinct persons
        hits = self.store.search(model, vec, limit=max(top_k, 1) * 12)
        candidates = best_per_person(hits)[:max(top_k, 1)]
        return decide(candidates, self.threshold if threshold is None else threshold, self.margin)

    def delete(self, model: str, person_id: int) -> None:
        self.check_model(model)
        self.store.delete_person(model, person_id)
