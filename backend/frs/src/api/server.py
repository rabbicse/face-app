import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import config
from api.routes.embeddings.route import embeddings_router

logging.basicConfig(level=os.getenv('FRS_LOG_LEVEL', 'INFO').upper(),
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')

version = "v1"
app = FastAPI(title="FRS embedding API", version=version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# client side embeddings: devices send 512-d vectors, the server only stores / searches them
app.include_router(embeddings_router, prefix=f'/api/{version}/embeddings')


@app.get(f'/api/{version}/health')
def health():
    return {'status': 'ok', 'embedding_models': config.EMBEDDING_MODELS}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("FR_PORT", 5000)))
