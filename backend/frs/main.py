"""
ASGI entrypoint of the API, importable from the backend/frs directory:

    uvicorn main:app --reload --port 5000

Also the entrypoint of the Vercel deployment (see [tool.vercel] in pyproject.toml).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from api.server import app  # noqa: E402

__all__ = ['app']
