import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependency_resolver import get_embedding_service
from api.services.embedding_service import EmbeddingService, EmbeddingValidationError, PersonMatch

logger = logging.getLogger(__name__)
embeddings_router = APIRouter()

# A 512-d embedding either as a JSON array or as base64 of little endian float32 (~2.7 KB).
Embedding = Union[List[float], str]


class PersonIn(BaseModel):
    person_id: int
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    age: Optional[int] = None
    city: Optional[str] = None
    country: Optional[str] = None
    address: Optional[str] = None


class HeadPoseIn(BaseModel):
    yaw: float
    pitch: float
    roll: float


class SampleIn(BaseModel):
    embedding: Embedding
    pose_label: Optional[str] = None
    pose: Optional[HeadPoseIn] = None
    quality: Optional[float] = None


class RegisterRequest(BaseModel):
    model: str
    person: PersonIn
    samples: List[SampleIn] = Field(min_length=1, max_length=64)


class IdentifyRequest(BaseModel):
    model: str
    embedding: Embedding
    top_k: int = Field(default=3, ge=1, le=20)
    threshold: Optional[float] = Field(default=None, ge=-1.0, le=1.0)


def _person(match: Optional[PersonMatch]) -> Optional[Dict[str, Any]]:
    if match is None:
        return None
    keys = ('person_id', 'name', 'email', 'phone', 'age', 'city', 'country', 'address')
    return {k: match.payload.get(k) for k in keys}


@embeddings_router.post('/register')
def register(req: RegisterRequest, service: EmbeddingService = Depends(get_embedding_service)):
    try:
        samples = [{'embedding': s.embedding, 'pose_label': s.pose_label,
                    'pose': s.pose.model_dump() if s.pose else None, 'quality': s.quality} for s in req.samples]
        info = service.register(req.model, req.person.model_dump(), samples)
        return {'status': 0, 'message': 'Face enrolled successfully!', **info}
    except EmbeddingValidationError as x:
        raise HTTPException(status_code=422, detail=str(x))


@embeddings_router.post('/identify')
def identify(req: IdentifyRequest, service: EmbeddingService = Depends(get_embedding_service)):
    try:
        result = service.identify(req.model, req.embedding, top_k=req.top_k, threshold=req.threshold)
    except EmbeddingValidationError as x:
        raise HTTPException(status_code=422, detail=str(x))
    if result.candidates:
        best = result.candidates[0]
        logger.info(f'identify: {result.reason} (best person {best.person_id}, score {best.score:.4f})')
    else:
        logger.info(f'identify: {result.reason}')
    return {
        'status': 0 if result.match else 2,
        'reason': result.reason,
        'score': round(result.match.score, 4) if result.match else None,
        'person': _person(result.match),
        'candidates': [{'person_id': c.person_id, 'score': round(c.score, 4)} for c in result.candidates],
    }


@embeddings_router.delete('/persons/{person_id}')
def delete_person(person_id: int, model: str, service: EmbeddingService = Depends(get_embedding_service)):
    try:
        service.delete(model, person_id)
    except EmbeddingValidationError as x:
        raise HTTPException(status_code=422, detail=str(x))
    return {'status': 0}
