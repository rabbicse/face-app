import logging

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.params import Depends

from api.dependency_resolver import get_face_service, get_vector_db_context
from api.services.face_service import FaceService
from utils.vision_utils.utils import convert_photo_to_bgr
from vector_db.qdrant_context import VectorDbContext

logger = logging.getLogger(__name__)
login_router = APIRouter()

@login_router.post("/")
async def login(
        photo: UploadFile = File(...),
        face_service: FaceService = Depends(get_face_service),
        vector_db_context: VectorDbContext = Depends(get_vector_db_context)):
    try:
        logger.info(f'Requesting login...')
        photo_data = await photo.read()
        frame = convert_photo_to_bgr(photo_data)

        logger.info('extracting embeddings...')
        emb = face_service.extract_embedding(frame)

        if isinstance(emb, int):
            return {"status": 2, "embedding": str(-3)}

        logger.info(f'searching face embedding to database...')
        result = vector_db_context.search_embedding(vector=emb)

        return {"status": 0,
                "result": result}
    except Exception as x:
        logger.error(f"Error when enrolling by image. Details: {x}")
        raise HTTPException(status_code=500, detail="Not a valid image!")