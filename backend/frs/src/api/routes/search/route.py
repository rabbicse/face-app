import base64
import json
import logging
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.params import Depends
from api.dependency_resolver import get_face_service, get_cache_service
from api.services.face_service import FaceService
from cache.cache_service import CacheService
from cache.redis_service import RedisCacheService
from utils.vision_utils.utils import convert_photo_to_bgr

logger = logging.getLogger(__name__)
search_router = APIRouter()


@search_router.post("/photo/exact")
async def exact_search_by_photo(image: UploadFile = File(...),
                                data: str = Form(...),
                                face_service: FaceService = Depends(get_face_service),
                                cache_service: RedisCacheService = Depends(get_cache_service)):
    try:
        photo_data = await image.read()
        frame = convert_photo_to_bgr(photo_data)

        emb = face_service.extract_embedding(frame)

        if isinstance(emb, int) and emb == -2:
            return {"status": 2, "score": str(-3)}

        person_info = json.loads(data)
        name = person_info["name"]

        # Retrieve embedding from Redis
        stored_emb = cache_service.search(name)
        score = face_service.match(emb, stored_emb)
        return {"status": 0, "score": str(score)}
    except Exception as x:
        logger.error(f"Error when matching by image. Details: {x}")
        raise HTTPException(status_code=500, detail=f"Unexpected exception while processing image! Details: {x}")

