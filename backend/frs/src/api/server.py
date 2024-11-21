import base64
import json
import os
import warnings
import numpy as np
import cv2

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import config
from utils.dnn_utils import dnn_converter
from face_handler import FaceHandler
from utils.vision_utils import log_utils
from utils.vision_utils.redis_handler import RedisHandler

warnings.filterwarnings("ignore")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = log_utils.LogUtils().get_logger(__name__)

face_handler = FaceHandler(det_model_path=config.DETECTION_MODEL_PATH,
                           det_model_tar=config.DETECTION_MODEL_TAR_PATH,
                           rec_model_path=config.RECOGNITION_MODEL_PATH)
redis_handler = RedisHandler()


@app.post("/detect/v1")
async def detect_v1(photo: bytes = File(...)):
    try:
        data = np.frombuffer(photo, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        faces = face_handler.detect_faces(frame)
        return {"status": "success", "faces": faces}
    except Exception as x:
        logger.error(f"Error when detecting faces by image. Details: {x}")
        raise HTTPException(status_code=400, detail="Not a valid image!")


@app.post("/embedding/v1")
async def extract_embedding_v1(photo: UploadFile = File(...)):
    try:
        photo_data = await photo.read()
        data = np.frombuffer(photo_data, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        emb = face_handler.extract_embedding(frame)

        if isinstance(emb, int) and emb == -2:
            return {"status": 2, "embedding": str(-3)}

        emb_b64 = base64.b64encode(emb.tobytes()).decode("ascii")
        return {"status": 0, "embedding": emb_b64}
    except Exception as x:
        logger.error(f"Error when recognizing by image. Details: {x}")
        raise HTTPException(status_code=400, detail="Not a valid image!")


# @app.post("/embedding/v2")
# async def extract_embedding_v2(photo: UploadFile = File(...)):
#     try:
#         photo_data = await photo.read()
#         data = np.frombuffer(photo_data, dtype=np.uint8)
#         frame = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
#         emb = face_handler.extract_embedding(frame)
#
#         if isinstance(emb, int) and emb == -2:
#             return {"status": 2, "embedding": str(-3)}
#
#         embedding = dnn_converter.encode_np_hex(emb)
#         return {"status": 0, "embedding": embedding}
#     except Exception as x:
#         logger.error(f"Error when recognizing by image. Details: {x}")
#         raise HTTPException(status_code=400, detail=f"Unexpected Exception: {x}")


@app.post("/enroll/v1")
async def enroll_v1(image: UploadFile = File(...), data: str = Form(...)):
    try:
        photo_data = await image.read()
        frame = np.frombuffer(photo_data, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_UNCHANGED)
        emb = face_handler.extract_embedding(frame)

        person_info = json.loads(data)
        name = person_info["name"]

        # Insert into Redis (or database)
        redis_handler.insert_data(name, emb)
        return {"status": 0, "embedding": emb.tolist()}
    except Exception as x:
        logger.error(f"Error when enrolling by image. Details: {x}")
        raise HTTPException(status_code=400, detail="Not a valid image!")


# @app.post("/match/v1")
# async def match_v1(image: UploadFile = File(...), data: str = Form(...)):
#     try:
#         photo_data = await image.read()
#         frame = np.frombuffer(photo_data, dtype=np.uint8)
#         frame = cv2.imdecode(frame, cv2.IMREAD_UNCHANGED)
#         emb = face_handler.extract_embedding(frame)
#
#         if isinstance(emb, int) and emb == -2:
#             return {"status": 2, "score": str(-3)}
#
#         person_info = json.loads(data)
#         name = person_info["name"]
#
#         # Retrieve embedding from Redis
#         stored_emb = redis_handler.search_data(name)
#         score = face_handler.match(emb, stored_emb)
#         return {"status": 0, "score": str(score)}
#     except Exception as x:
#         logger.error(f"Error when matching by image. Details: {x}")
#         raise HTTPException(status_code=400, detail="Not a valid image!")


@app.post("/match/v2")
async def match_v2(photo: UploadFile = File(...), embeddings: str = Form(...)):
    try:
        photo_data = await photo.read()
        frame = np.frombuffer(photo_data, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_UNCHANGED)

        if frame.shape[-1] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if len(frame.shape) == 2:  # Grayscale image
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        emb = face_handler.extract_embedding(frame)

        if isinstance(emb, int) and emb == -2:
            return {"status": 2, "score": str(-3)}

        # embeddings_data = json.loads(embeddings)
        # embedding = dnn_converter.decode_np_hex(embeddings_data["embedding"])
        # embedding = dnn_converter.decode_np(embeddings)
        b64_bytes = embeddings.encode('ascii')
        data_bytes = base64.decodebytes(b64_bytes)
        embedding = np.frombuffer(data_bytes, dtype=np.float32)

        score = face_handler.match(emb, embedding)
        return {"status": 0, "score": str(score)}
    except Exception as x:
        logger.error(f"Error when matching by image. Details: {x}")
        raise HTTPException(status_code=400, detail="Unexpected Exception!")


@app.post("/test/v1")
async def test_v1(data: str = Form(...)):
    try:
        return {"status": 0, "score": data}
    except Exception as x:
        logger.error(f"Error when testing. Details: {x}")
        raise HTTPException(status_code=400, detail="Error occurred!")


@app.get("/test/v2")
async def test_v2():
    try:
        return {"status": 0, "score": 0}
    except Exception as x:
        logger.error(f"Error when testing. Details: {x}")
        raise HTTPException(status_code=400, detail="Error occurred!")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("FR_PORT", 5000)))
