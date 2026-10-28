"""
Reference (ground-truth) face pipeline in plain Python + ONNX Runtime.

Every client (React web, Flutter Android/iOS) must produce the SAME 512-d
vector as this file for the same face, otherwise vectors enrolled on one
device won't match identifications from another. Use this to:

  * check parity of the web / mobile implementations (examples/python/parity_check.py)
  * embed images from disk (examples/python/embed_image.py)
  * compare two photos (examples/python/compare_faces.py)

Pipeline (identical on every platform):
  1. detect face + 5 landmarks (here: SCRFD det_500m; web: MediaPipe; mobile: ML Kit)
  2. similarity-transform (Umeyama) the 5 points onto the ArcFace 112x112 template
  3. warp to 112x112 RGB, normalise (x - 127.5) / 127.5, NCHW
  4. run w600k_mbf.onnx (MobileFaceNet / ArcFace, WebFace600K) -> 512 floats
  5. L2-normalise -> cosine similarity == dot product
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

# ArcFace canonical 5-point template for a 112x112 crop
# (left eye, right eye, nose tip, left mouth corner, right mouth corner)
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS = os.path.abspath(os.path.join(HERE, "..", "..", "models"))


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation + uniform scale + translation)
    mapping src -> dst. Returns a 2x3 affine matrix. Same maths as
    skimage.transform.SimilarityTransform.estimate and as the TS / Dart ports."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = dst_c.T @ src_c / n
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1
    r = u @ np.diag(d) @ vt
    var_src = (src_c**2).sum() / n
    scale = (s * d).sum() / var_src
    t = dst_mean - scale * r @ src_mean
    m = np.zeros((2, 3))
    m[:, :2] = scale * r
    m[:, 2] = t
    return m


def align_face(img_bgr: np.ndarray, kps5: np.ndarray, size: int = 112) -> np.ndarray:
    """Warp the face to the canonical 112x112 crop. Returns RGB uint8."""
    m = umeyama(kps5, ARCFACE_TEMPLATE)
    warped = cv2.warpAffine(img_bgr, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=0.0)
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)


@dataclass
class Face:
    bbox: np.ndarray  # x1, y1, x2, y2
    kps: np.ndarray  # 5x2
    score: float


class SCRFD:
    """Minimal SCRFD (det_500m.onnx) decoder: boxes + 5 keypoints."""

    def __init__(self, model_path: str, input_size: int = 640, thresh: float = 0.5):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.input_size = input_size
        self.thresh = thresh
        self.strides = [8, 16, 32]
        self.num_anchors = 2
        self._anchor_cache: dict = {}

    def detect(self, img_bgr: np.ndarray, max_num: int = 0) -> list[Face]:
        h, w = img_bgr.shape[:2]
        scale = self.input_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(img_bgr, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (self.input_size, self.input_size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.sess.run(None, {self.input_name: blob})

        scores_all, boxes_all, kps_all = [], [], []
        fmc = len(self.strides)
        for idx, stride in enumerate(self.strides):
            scores = outs[idx][:, 0]
            bbox_preds = outs[idx + fmc] * stride
            kps_preds = outs[idx + fmc * 2] * stride
            fh = fw = self.input_size // stride
            key = (fh, fw, stride)
            if key not in self._anchor_cache:
                ac = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32)
                ac = (ac * stride).reshape(-1, 2)
                ac = np.stack([ac] * self.num_anchors, axis=1).reshape(-1, 2)
                self._anchor_cache[key] = ac
            centers = self._anchor_cache[key]
            pos = np.where(scores >= self.thresh)[0]
            x1 = centers[:, 0] - bbox_preds[:, 0]
            y1 = centers[:, 1] - bbox_preds[:, 1]
            x2 = centers[:, 0] + bbox_preds[:, 2]
            y2 = centers[:, 1] + bbox_preds[:, 3]
            boxes = np.stack([x1, y1, x2, y2], axis=-1)
            kps = np.zeros((kps_preds.shape[0], 5, 2), dtype=np.float32)
            for k in range(5):
                kps[:, k, 0] = centers[:, 0] + kps_preds[:, 2 * k]
                kps[:, k, 1] = centers[:, 1] + kps_preds[:, 2 * k + 1]
            scores_all.append(scores[pos])
            boxes_all.append(boxes[pos])
            kps_all.append(kps[pos])

        scores = np.concatenate(scores_all)
        boxes = np.concatenate(boxes_all) / scale
        kps = np.concatenate(kps_all) / scale
        order = scores.argsort()[::-1]
        keep = self._nms(boxes[order], scores[order], 0.4)
        faces = [Face(boxes[order][i], kps[order][i], float(scores[order][i])) for i in keep]
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return faces[:max_num] if max_num else faces

    @staticmethod
    def _nms(boxes, scores, thr):
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.arange(len(scores))
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(iou <= thr)[0] + 1]
        return keep


class Embedder:
    def __init__(self, model_path: str):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    @staticmethod
    def preprocess(aligned_rgb: np.ndarray) -> np.ndarray:
        x = (aligned_rgb.astype(np.float32) - 127.5) / 127.5
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def embed_aligned(self, aligned_rgb: np.ndarray) -> np.ndarray:
        out = self.sess.run(None, {self.input_name: self.preprocess(aligned_rgb)})[0][0]
        return (out / np.linalg.norm(out)).astype(np.float32)


class FacePipeline:
    def __init__(self, models_dir: str = DEFAULT_MODELS, embed_model: str = "w600k_mbf.onnx"):
        self.detector = SCRFD(os.path.join(models_dir, "det_500m.onnx"))
        self.embedder = Embedder(os.path.join(models_dir, embed_model))

    def embed_image(self, img_bgr: np.ndarray, max_faces: int = 1):
        faces = self.detector.detect(img_bgr, max_num=max_faces)
        return [(f, self.embedder.embed_aligned(align_face(img_bgr, f.kps))) for f in faces]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
