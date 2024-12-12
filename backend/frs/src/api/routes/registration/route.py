import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.params import Depends

from api.dependency_resolver import get_face_service, get_vector_db_context
from api.model.person import Person
from api.services.face_service import FaceService
from utils.vision_utils.utils import convert_photo_to_bgr
from vector_db.qdrant_context import VectorDbContext

logger = logging.getLogger(__name__)
register_router = APIRouter()


@register_router.post("/")
async def register(
        person: Person = Depends(Person.as_form), #Person = Form(...),
        photo: UploadFile = File(...),
        face_service: FaceService = Depends(get_face_service),
        vector_db_context: VectorDbContext = Depends(get_vector_db_context)):
    try:
        logger.info(f'Requesting registration...')
        photo_data = await photo.read()
        frame = convert_photo_to_bgr(photo_data)

        logger.info('extracting embeddings...')
        emb = face_service.extract_embedding(frame)

        if isinstance(emb, int):
            return {"status": 2, "embedding": str(-3)}

        logger.info(f'inserting vector to database...')
        vector_db_context.upsert_embedding(vector=emb, person=person)

        return {"status": 0, "embedding": emb.tolist()}
    except Exception as x:
        logger.error(f"Error when enrolling by image. Details: {x}")
        raise HTTPException(status_code=500, detail="Not a valid image!")
