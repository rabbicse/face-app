import os
import warnings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.dnn.route import dnn_router
from api.routes.login.route import login_router
from api.routes.registration.route import register_router
from api.routes.search.route import search_router
from utils.vision_utils import log_utils

warnings.filterwarnings("ignore")
logger = log_utils.LogUtils().get_logger(__name__)

version = "v1"
app = FastAPI(
    version=version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(register_router, prefix=f'/api/{version}/register')
app.include_router(login_router, prefix=f'/api/{version}/login')
app.include_router(dnn_router, prefix=f'/api/{version}/dnn')
app.include_router(search_router, prefix=f'/api/{version}/search')



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("FR_PORT", 5000)))
