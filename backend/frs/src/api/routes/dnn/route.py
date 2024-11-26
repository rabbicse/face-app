import base64
import json
import logging
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.params import Depends
from api.dependency_resolver import get_face_service
from api.services.face_service import FaceService
from utils.vision_utils.utils import convert_photo_to_bgr

logger = logging.getLogger(__name__)
dnn_router = APIRouter()


@dnn_router.post("/detect")
async def detect(photo: UploadFile = File(...),
                 face_service: FaceService = Depends(get_face_service)):
    try:
        photo_data = await photo.read()
        frame = convert_photo_to_bgr(photo_data)

        faces = face_service.detect_faces(frame)
        return {"status": "success", "faces": faces}
    except Exception as x:
        logger.error(f"Error when detecting faces by image. Details: {x}")
        raise HTTPException(status_code=400, detail="Not a valid image!")


@dnn_router.post("/extract-embedding")
async def extract_embedding(photo: UploadFile = File(...),
                            face_service: FaceService = Depends(get_face_service)):
    try:
        photo_data = await photo.read()
        frame = convert_photo_to_bgr(photo_data)

        emb = face_service.extract_embedding(frame)

        if isinstance(emb, int) and emb == -2:
            return {"status": 2, "embedding": str(-3)}

        emb_b64 = base64.b64encode(emb.tobytes()).decode("ascii")
        return {"status": 0, "embedding": emb_b64}
    except Exception as x:
        logger.error(f"Error when recognizing by image. Details: {x}")
        raise HTTPException(status_code=400, detail="Not a valid image!")


@dnn_router.post("/match/photo-embedding")
async def match_photo_with_embedding(photo: UploadFile = File(...),
                                     embeddings: str = Form(...),
                                     face_service: FaceService = Depends(get_face_service)):
    try:
        photo_data = await photo.read()
        frame = convert_photo_to_bgr(photo_data)

        emb = face_service.extract_embedding(frame)

        if isinstance(emb, int) and emb == -2:
            return {"status": 2, "score": str(-3)}

        b64_bytes = embeddings.encode('ascii')
        data_bytes = base64.decodebytes(b64_bytes)
        embedding = np.frombuffer(data_bytes, dtype=np.float32)

        score = face_service.match(emb, embedding)
        return {"status": 0, "score": str(score)}
    except Exception as x:
        logger.error(f"Error when matching by image. Details: {x}")
        raise HTTPException(status_code=400, detail="Unexpected Exception!")


@dnn_router.post("/match/embeddings")
async def match_embeddings(embedding_one: str = Form(...),
                           embedding_two: str = Form(...),
                           face_service: FaceService = Depends(get_face_service)):
    try:
        b64_bytes = embedding_one.encode('ascii')
        data_bytes = base64.decodebytes(b64_bytes)
        emb_one = np.frombuffer(data_bytes, dtype=np.float32)

        b64_bytes = embedding_two.encode('ascii')
        data_bytes = base64.decodebytes(b64_bytes)
        emb_two = np.frombuffer(data_bytes, dtype=np.float32)

        score = face_service.match(emb_one, emb_two)
        return {"status": 0, "score": str(score)}
    except Exception as x:
        logger.error(f"Error when matching by image. Details: {x}")
        raise HTTPException(status_code=400, detail="Unexpected Exception!")
