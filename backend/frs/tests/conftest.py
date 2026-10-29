import os

import pytest

# lightweight API for tests: in-memory Qdrant, no torch / model weights
os.environ.setdefault('QDRANT_LOCATION', ':memory:')
os.environ.setdefault('FRS_ENABLE_SERVER_INFERENCE', 'false')

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
TEST_VECTORS = os.path.join(REPO_ROOT, 'shared', 'test-vectors', 'face-pipeline.json')


@pytest.fixture(scope='session')
def test_vectors():
    import json

    with open(TEST_VECTORS) as f:
        return json.load(f)
