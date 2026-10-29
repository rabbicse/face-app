import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, FilterSelector, MatchValue, PointStruct
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from utils.vision_utils import log_utils

logger = log_utils.LogUtils().get_logger(__name__)


@dataclass
class StoredVector:
    vector: np.ndarray
    kind: str  # 'sample' or 'template'
    pose_label: Optional[str] = None
    pose: Optional[Dict[str, float]] = None
    quality: Optional[float] = None


@dataclass
class Hit:
    person_id: int
    score: float
    payload: Dict[str, Any]


class EmbeddingStore:
    """
    Qdrant backed store of client generated face embeddings. One collection per embedding model,
    several points per person (one per registered pose sample plus a fused template).
    """

    def __init__(self, client: QdrantClient, collection_prefix: str, dim: int):
        self.client = client
        self.collection_prefix = collection_prefix
        self.dim = dim
        self._ready: set = set()

    def collection_name(self, model: str) -> str:
        return f'{self.collection_prefix}{model}'

    def _ensure_collection(self, model: str) -> str:
        name = self.collection_name(model)
        if name in self._ready:
            return name
        if not self.client.collection_exists(name):
            self.client.create_collection(collection_name=name,
                                          vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE))
            try:
                self.client.create_payload_index(name, field_name='person_id',
                                                 field_schema=PayloadSchemaType.INTEGER)
            except Exception as x:  # local / in-memory mode does not support indexes
                logger.debug(f'payload index not created for {name}: {x}')
        self._ready.add(name)
        return name

    @staticmethod
    def _person_filter(person_id: int) -> Filter:
        return Filter(must=[FieldCondition(key='person_id', match=MatchValue(value=person_id))])

    def delete_person(self, model: str, person_id: int) -> None:
        name = self._ensure_collection(model)
        self.client.delete(collection_name=name,
                           points_selector=FilterSelector(filter=self._person_filter(person_id)),
                           wait=True)

    def replace_person(self, model: str, person: Dict[str, Any], vectors: Sequence[StoredVector]) -> int:
        """Remove previous vectors of the person and store the new ones. Returns the number of points."""
        name = self._ensure_collection(model)
        person_id = int(person['person_id'])
        self.delete_person(model, person_id)

        points = []
        for v in vectors:
            payload = dict(person)
            payload.update({'kind': v.kind, 'pose_label': v.pose_label, 'pose': v.pose,
                            'quality': v.quality, 'model': model})
            points.append(PointStruct(id=str(uuid.uuid4()), vector=v.vector.tolist(), payload=payload))
        self.client.upsert(collection_name=name, points=points, wait=True)
        logger.info(f'stored {len(points)} vectors for person {person_id} in {name}')
        return len(points)

    def search(self, model: str, vector: np.ndarray, limit: int) -> List[Hit]:
        name = self._ensure_collection(model)
        points = self.client.query_points(collection_name=name, query=vector.tolist(),
                                          with_payload=True, limit=limit).points
        return [Hit(person_id=int(p.payload['person_id']), score=float(p.score), payload=p.payload)
                for p in points]

    def count_person(self, model: str, person_id: int) -> int:
        name = self._ensure_collection(model)
        return self.client.count(collection_name=name, count_filter=self._person_filter(person_id),
                                 exact=True).count
