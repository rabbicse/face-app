import numpy as np
import pytest
from qdrant_client import QdrantClient

from api.services.embedding_service import (EmbeddingService, EmbeddingValidationError, PersonMatch, decide,
                                            parse_embedding)
from face_pipeline.embedding import encode_base64
from vector_db.embedding_store import EmbeddingStore

DIM = 512
MODEL = 'w600k_mbf'


def unit(v):
    return v / np.linalg.norm(v)


def identity(seed):
    return unit(np.random.default_rng(seed).normal(size=DIM)).astype(np.float32)


def noisy(base, seed, noise=0.03):
    return unit(base + np.random.default_rng(seed).normal(scale=noise, size=DIM)).astype(np.float32)


@pytest.fixture()
def service():
    store = EmbeddingStore(QdrantClient(location=':memory:'), collection_prefix='test_', dim=DIM)
    return EmbeddingService(store, dim=DIM, models=[MODEL], threshold=0.45, margin=0.05)


def person(pid, name):
    return {'person_id': pid, 'name': name, 'email': f'{name}@example.com'}


def test_parse_embedding_accepts_list_and_base64():
    v = identity(1)
    np.testing.assert_allclose(parse_embedding(v.tolist(), DIM), v, atol=1e-6)
    np.testing.assert_allclose(parse_embedding(encode_base64(v * 7), DIM), v, atol=1e-6)


@pytest.mark.parametrize('bad', [[0.1] * 10, [0.0] * DIM, [float('nan')] * DIM, 'not-base64!'])
def test_parse_embedding_rejects_bad_input(bad):
    with pytest.raises(EmbeddingValidationError):
        parse_embedding(bad, DIM)


def test_decide_threshold_and_margin():
    a = PersonMatch(1, 0.80, {})
    b = PersonMatch(2, 0.77, {})
    c = PersonMatch(3, 0.30, {})
    assert decide([], 0.45, 0.05).reason == 'no_enrolled_faces'
    assert decide([c], 0.45, 0.05).reason == 'below_threshold'
    assert decide([a, b], 0.45, 0.05).reason == 'ambiguous'
    assert decide([a, c], 0.45, 0.05).match is a


def test_register_and_identify(service):
    alice, bob = identity(10), identity(20)
    service.register(MODEL, person(1, 'alice'), [{'embedding': noisy(alice, i)} for i in range(5)])
    service.register(MODEL, person(2, 'bob'), [{'embedding': noisy(bob, i + 100)} for i in range(5)])

    res = service.identify(MODEL, noisy(alice, 999))
    assert res.match is not None and res.match.person_id == 1
    assert res.match.payload['name'] == 'alice'
    assert [c.person_id for c in res.candidates][:2] == [1, 2]

    stranger = service.identify(MODEL, identity(30))
    assert stranger.match is None and stranger.reason == 'below_threshold'


def test_reregistration_replaces_previous_vectors(service):
    service.register(MODEL, person(1, 'alice'), [{'embedding': noisy(identity(10), i)} for i in range(3)])
    service.register(MODEL, person(1, 'alice'), [{'embedding': noisy(identity(11), i)} for i in range(2)])
    assert service.store.count_person(MODEL, 1) == 3  # 2 samples + template
    assert service.identify(MODEL, identity(10)).match is None
    assert service.identify(MODEL, identity(11)).match.person_id == 1


def test_register_rejects_mixed_identities(service):
    samples = [{'embedding': identity(1)}, {'embedding': identity(2)}, {'embedding': identity(3)}]
    with pytest.raises(EmbeddingValidationError):
        service.register(MODEL, person(1, 'x'), samples)


def test_unknown_model_rejected(service):
    with pytest.raises(EmbeddingValidationError):
        service.identify('r100', identity(1))
