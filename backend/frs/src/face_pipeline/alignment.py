from typing import Sequence

import numpy as np

# ArcFace / InsightFace 5-point template for a 112x112 crop (x offset +8 already applied).
# Order (in *image* coordinates of a non-mirrored frame):
#   0: eye on the image-left, 1: eye on the image-right, 2: nose tip,
#   3: mouth corner on the image-left, 4: mouth corner on the image-right
ARCFACE_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float64)

# Mid point between the mouth corners, used when a detector only reports a mouth centre
# (e.g. BlazeFace / MediaPipe short range detector).
ARCFACE_MOUTH_CENTER_112 = ARCFACE_TEMPLATE_112[3:5].mean(axis=0)


def estimate_similarity_transform(src: Sequence[Sequence[float]],
                                  dst: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Least squares, non-reflective similarity transform (rotation + uniform scale + translation)
    mapping ``src`` points onto ``dst`` points. Closed form of Umeyama's method for 2D, identical
    to ``skimage.transform.SimilarityTransform.estimate`` for non-degenerate face landmarks.

    :return: 2x3 affine matrix ``M`` such that ``dst ~= M @ [x, y, 1]``
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or src.shape[0] < 2:
        raise ValueError(f"expected two (N>=2, 2) point sets, got {src.shape} and {dst.shape}")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    s = src - src_mean
    d = dst - dst_mean

    denom = float((s ** 2).sum())
    if denom < 1e-12:
        raise ValueError("source points are degenerate")

    a = float((s[:, 0] * d[:, 0] + s[:, 1] * d[:, 1]).sum()) / denom
    b = float((s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]).sum()) / denom

    tx = dst_mean[0] - (a * src_mean[0] - b * src_mean[1])
    ty = dst_mean[1] - (b * src_mean[0] + a * src_mean[1])
    return np.array([[a, -b, tx],
                     [b, a, ty]], dtype=np.float64)


def alignment_matrix(landmarks: Sequence[Sequence[float]], image_size: int = 112) -> np.ndarray:
    """
    :param landmarks: 5 points (see ``ARCFACE_TEMPLATE_112`` for the order) or 4 points
                      (left eye, right eye, nose, mouth centre) in source pixel coordinates.
    """
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.shape == (5, 2):
        template = ARCFACE_TEMPLATE_112
    elif pts.shape == (4, 2):
        template = np.vstack([ARCFACE_TEMPLATE_112[:3], ARCFACE_MOUTH_CENTER_112])
    else:
        raise ValueError(f"expected 5 or 4 landmarks, got shape {pts.shape}")
    return estimate_similarity_transform(pts, template * (image_size / 112.0))


def warp_face(image: np.ndarray, landmarks: Sequence[Sequence[float]], image_size: int = 112) -> np.ndarray:
    """Return the aligned ``image_size`` x ``image_size`` face crop (same channel order as ``image``)."""
    import cv2

    m = alignment_matrix(landmarks, image_size)
    return cv2.warpAffine(image, m, (image_size, image_size), flags=cv2.INTER_LINEAR, borderValue=0.0)
