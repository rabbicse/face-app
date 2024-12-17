import json
import logging

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import Distance, VectorParams

from api.model.person import Person
from utils.vision_utils.singleton_decorator import SingletonDecorator

logger = logging.getLogger(__name__)


@SingletonDecorator
class VectorDbContext:
    def __init__(self, host: str = 'localhost', port: int = 6333, collection_name='face_collection'):
        # initiate qdrant client
        self.client = QdrantClient(url=f"http://{host}:{port}")

        # set collection name
        self.collection_name = collection_name

        # if not exists
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.DOT),
            )

    def upsert(self, vector: np.ndarray, name: str, id: int):
        payload = {"name": name}
        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=[
                PointStruct(id=id, vector=vector, payload=payload),
            ],
        )
        print(operation_info)

    def upsert_embedding(self, vector: np.ndarray, person: Person):
        payload = dict(person)
        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=[
                PointStruct(id=person.person_id, vector=vector, payload=payload),
            ],
        )
        print(operation_info)

    def search_embedding(self, vector: np.ndarray):
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            with_payload=False,
            limit=10
        ).points

        print(search_result)
