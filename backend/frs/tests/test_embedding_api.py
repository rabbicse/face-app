import numpy as np
import pytest
from fastapi.testclient import TestClient

from face_pipeline.embedding import encode_base64

MODEL = 'w600k_mbf'


@pytest.fixture(scope='module')
def client():
    from api.server import app

    return TestClient(app)


def vec(seed, base=None):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=512) if base is None else base + rng.normal(scale=0.03, size=512)
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_health(client):
    r = client.get('/api/v1/health')
    assert r.status_code == 200 and MODEL in r.json()['embedding_models']


def test_register_identify_delete_flow(client):
    base = vec(1)
    payload = {
        'model': MODEL,
        'person': {'person_id': 42, 'name': 'Mehedi', 'email': 'm@example.com'},
        'samples': [
            {'embedding': encode_base64(vec(2, base)), 'pose_label': 'frontal',
             'pose': {'yaw': 1, 'pitch': 0, 'roll': 2}, 'quality': 0.9},
            {'embedding': vec(3, base).tolist(), 'pose_label': 'turn_left',
             'pose': {'yaw': 22, 'pitch': 1, 'roll': 0}},
        ],
    }
    r = client.post('/api/v1/embeddings/register', json=payload)
    assert r.status_code == 200, r.text
    assert r.json()['status'] == 0 and r.json()['stored_vectors'] == 3

    r = client.post('/api/v1/embeddings/identify', json={'model': MODEL, 'embedding': encode_base64(vec(4, base))})
    body = r.json()
    assert body['status'] == 0 and body['reason'] == 'matched'
    assert body['person']['person_id'] == 42 and body['person']['name'] == 'Mehedi'
    assert body['score'] > 0.6

    r = client.post('/api/v1/embeddings/identify', json={'model': MODEL, 'embedding': vec(5).tolist()})
    assert r.json()['status'] == 2 and r.json()['person'] is None

    assert client.delete('/api/v1/embeddings/persons/42', params={'model': MODEL}).json()['status'] == 0
    r = client.post('/api/v1/embeddings/identify', json={'model': MODEL, 'embedding': encode_base64(vec(4, base))})
    assert r.json()['status'] == 2


@pytest.mark.parametrize('body', [
    {'model': MODEL, 'embedding': [0.1, 0.2]},
    {'model': 'unknown', 'embedding': [0.1] * 512},
    {'model': MODEL},
])
def test_identify_validation(client, body):
    assert client.post('/api/v1/embeddings/identify', json=body).status_code == 422
