import os
import warnings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import config
from api.routes.embeddings.route import embeddings_router
from utils.vision_utils import log_utils

warnings.filterwarnings("ignore")
logger = log_utils.LogUtils().get_logger(__name__)

version = "v1"
app = FastAPI(
    version=version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# client side embeddings: devices send 512-d vectors, the server only stores / searches them
app.include_router(embeddings_router, prefix=f'/api/{version}/embeddings')

if config.ENABLE_SERVER_INFERENCE:
    try:
        # legacy image based endpoints, they need torch and the RetinaFace / ArcFace weights
        from api.routes.dnn.route import dnn_router
        from api.routes.login.route import login_router
        from api.routes.registration.route import register_router
        from api.routes.search.route import search_router

        app.include_router(register_router, prefix=f'/api/{version}/register')
        app.include_router(login_router, prefix=f'/api/{version}/login')
        app.include_router(dnn_router, prefix=f'/api/{version}/dnn')
        app.include_router(search_router, prefix=f'/api/{version}/search')
    except ImportError as x:
        logger.warning(f'Server side inference disabled, missing dependency: {x}')


@app.get(f'/api/{version}/health')
def health():
    return {'status': 'ok', 'embedding_models': config.EMBEDDING_MODELS, 'server_inference': config.ENABLE_SERVER_INFERENCE}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("FR_PORT", 5000)))
