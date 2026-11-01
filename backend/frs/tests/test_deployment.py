import os
import subprocess
import sys

import pytest

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def test_main_entrypoint_imports_from_backend_root():
    # what `uvicorn main:app` and Vercel's runtime do: import main.py with backend/frs as the working dir
    code = 'import main; print(sorted(r.path for r in main.app.routes if r.path.startswith("/api")))'
    env = {**os.environ, 'QDRANT_LOCATION': ':memory:', 'FRS_ENABLE_SERVER_INFERENCE': 'false', 'PYTHONPATH': ''}
    out = subprocess.run([sys.executable, '-c', code], cwd=BACKEND, env=env, capture_output=True, text=True, check=True)
    assert '/api/v1/embeddings/identify' in out.stdout and '/api/v1/health' in out.stdout


def test_pyproject_declares_vercel_entrypoint():
    tomllib = pytest.importorskip('tomllib')
    with open(os.path.join(BACKEND, 'pyproject.toml'), 'rb') as f:
        data = tomllib.load(f)
    assert data['tool']['vercel']['entrypoint'] == 'main:app'
    deps = ' '.join(data['project']['dependencies'])
    assert 'torch' not in deps and 'opencv' not in deps  # stays small enough for a serverless function


@pytest.mark.parametrize('env,expected', [
    ({'QDRANT_URL': 'https://abc.cloud.qdrant.io:6333', 'QDRANT_API_KEY': 'secret'},
     {'url': 'https://abc.cloud.qdrant.io:6333', 'api_key': 'secret'}),
    ({'QDRANT_HOST': 'db', 'QDRANT_PORT': '7000'}, {'url': 'http://db:7000'}),
])
def test_qdrant_client_selection(monkeypatch, env, expected):
    from api import config, dependency_resolver

    monkeypatch.setattr(config, 'QDRANT_LOCATION', None)
    monkeypatch.setattr(config, 'QDRANT_URL', env.get('QDRANT_URL'))
    monkeypatch.setattr(config, 'QDRANT_API_KEY', env.get('QDRANT_API_KEY'))
    monkeypatch.setattr(config, 'QDRANT_HOST', env.get('QDRANT_HOST', 'localhost'))
    monkeypatch.setattr(config, 'QDRANT_PORT', int(env.get('QDRANT_PORT', 6333)))
    calls = []
    monkeypatch.setattr('qdrant_client.QdrantClient', lambda **kw: calls.append(kw) or object())
    dependency_resolver.get_qdrant_client.cache_clear()
    try:
        dependency_resolver.get_qdrant_client()
    finally:
        dependency_resolver.get_qdrant_client.cache_clear()
    assert calls == [expected]
