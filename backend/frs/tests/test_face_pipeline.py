import numpy as np
import pytest

from face_pipeline.alignment import ARCFACE_TEMPLATE_112, alignment_matrix, estimate_similarity_transform, warp_face
from face_pipeline.embedding import (cosine_similarity, decode_base64, encode_base64, fuse_embeddings,
                                     l2_normalize)
from face_pipeline.pose import HeadPose, classify_pose, pose_from_landmarks_3d
from face_pipeline.preprocess import to_model_input


def test_identity_transform_for_template_points():
    m = alignment_matrix(ARCFACE_TEMPLATE_112)
    np.testing.assert_allclose(m, [[1, 0, 0], [0, 1, 0]], atol=1e-9)


def test_similarity_transform_matches_skimage():
    skimage = pytest.importorskip('skimage.transform')
    rng = np.random.default_rng(0)
    for _ in range(20):
        src = ARCFACE_TEMPLATE_112 * rng.uniform(0.5, 3) + rng.uniform(0, 400, size=2) + rng.normal(0, 3, (5, 2))
        tform = skimage.SimilarityTransform()
        tform.estimate(src, ARCFACE_TEMPLATE_112)
        np.testing.assert_allclose(estimate_similarity_transform(src, ARCFACE_TEMPLATE_112),
                                   tform.params[:2], atol=1e-6)


def test_alignment_recovers_known_transform():
    angle = np.radians(17)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    src = (ARCFACE_TEMPLATE_112 @ rot.T) * 2.5 + [120, 80]
    mapped = np.hstack([src, np.ones((5, 1))]) @ alignment_matrix(src).T
    np.testing.assert_allclose(mapped, ARCFACE_TEMPLATE_112, atol=1e-6)


def test_four_point_alignment_uses_mouth_centre():
    pts = np.vstack([ARCFACE_TEMPLATE_112[:3], ARCFACE_TEMPLATE_112[3:].mean(0)]) * 2 + 10
    m = alignment_matrix(pts)
    np.testing.assert_allclose(m, [[0.5, 0, -5], [0, 0.5, -5]], atol=1e-9)


def test_degenerate_points_raise():
    with pytest.raises(ValueError):
        estimate_similarity_transform([[1, 1]] * 5, ARCFACE_TEMPLATE_112)
    with pytest.raises(ValueError):
        alignment_matrix(np.zeros((3, 2)))


def test_warp_face_output_shape():
    pytest.importorskip('cv2')
    img = np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    pts = ARCFACE_TEMPLATE_112 * 2 + [200, 100]
    assert warp_face(img, pts).shape == (112, 112, 3)


def test_preprocess_layout_and_range():
    rgb = np.zeros((112, 112, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    x = to_model_input(rgb)
    assert x.shape == (1, 3, 112, 112) and x.dtype == np.float32
    assert np.all(x[0, 0] == 1.0) and np.all(x[0, 1] == -1.0)


def test_embedding_helpers():
    a = np.array([3.0, 4.0])
    np.testing.assert_allclose(l2_normalize(a), [0.6, 0.8], atol=1e-7)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0)
    assert cosine_similarity([1, 1], [2, 2]) == pytest.approx(1)
    np.testing.assert_allclose(fuse_embeddings([[1, 0], [0, 1]]), [2 ** -0.5, 2 ** -0.5], atol=1e-7)
    v = np.arange(512, dtype=np.float32)
    np.testing.assert_array_equal(decode_base64(encode_base64(v)), v)
    with pytest.raises(ValueError):
        l2_normalize([0, 0])


@pytest.mark.parametrize('pose,label', [
    (HeadPose(0, 0, 0), 'frontal'),
    (HeadPose(25, 0, 0), 'turn_left'),
    (HeadPose(-25, 0, 0), 'turn_right'),
    (HeadPose(0, 20, 0), 'look_up'),
    (HeadPose(0, -20, 0), 'look_down'),
    (HeadPose(60, 0, 0), None),
])
def test_classify_pose(pose, label):
    assert classify_pose(pose) == label


def test_pose_frontal_points():
    p = pose_from_landmarks_3d([-30, 0, 0], [30, 0, 0], [0, -60, 0], [0, 70, 0])
    assert (p.yaw, p.pitch, p.roll) == pytest.approx((0, 0, 0), abs=1e-9)


def test_shared_vectors_are_up_to_date(test_vectors):
    from generate_test_vectors import build

    assert build() == test_vectors, 'run: python tools/generate_test_vectors.py'
