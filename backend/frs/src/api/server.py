import base64
import json
import os
import warnings
import numpy as np
import cv2

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import config
from api.routes.dnn.route import dnn_router
from api.routes.registration.route import register_router
from api.routes.search.route import search_router
from api.services.face_service import FaceService
from utils.vision_utils import log_utils
from cache.redis_service import RedisCacheService

warnings.filterwarnings("ignore")
logger = log_utils.LogUtils().get_logger(__name__)

version = "v1"
app = FastAPI(
    version=version
)

app.include_router(register_router, prefix=f'/api/{version}/register')
app.include_router(dnn_router, prefix=f'/api/{version}/dnn')
app.include_router(search_router, prefix=f'/api/{version}/search')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# face_handler = FaceService(det_model_path=config.DETECTION_MODEL_PATH,
#                            det_model_tar=config.DETECTION_MODEL_TAR_PATH,
#                            det_network=config.DETECTION_NETWORK,
#                            rec_model_path=config.RECOGNITION_MODEL_PATH,
#                            rec_network=config.RECOGNITION_NETWORK,
#                            device=config.DEVICE)
# redis_handler = RedisCacheService()



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("FR_PORT", 5000)))
