import json
import logging
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.params import Depends
from api.dependency_resolver import get_face_service
from api.services.face_service import FaceService

logger = logging.getLogger(__name__)
register_router = APIRouter()

@register_router.post("/")
async def register(image: UploadFile = File(...),
                   data: str = Form(...),
                   service: FaceService = Depends(get_face_service)):
    try:
        photo_data = await image.read()
        frame = np.frombuffer(photo_data, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_UNCHANGED)

        if frame.shape[-1] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if len(frame.shape) == 2:  # Grayscale image
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        emb = service.extract_embedding(frame)

        person_info = json.loads(data)
        name = person_info["name"]

        # Insert into Redis (or database) todo:
        # redis_handler.insert_data(name, emb)
        return {"status": 0, "embedding": emb.tolist()}
    except Exception as x:
        logger.error(f"Error when enrolling by image. Details: {x}")
        raise HTTPException(status_code=500, detail="Not a valid image!")