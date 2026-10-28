"""
Generate reference fixtures used by the TypeScript (vitest) and Dart parity tests.

    python examples/python/make_test_fixtures.py

Writes frontend/frs_react/tests/fixtures/reference.json and
mobile/face_app/test/fixtures/reference.json (same content).
"""
import base64
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from face_ref import ARCFACE_TEMPLATE, FacePipeline, align_face, umeyama  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMAGES = sorted(glob.glob(os.path.join(ROOT, "frontend/web/public/bbt*.jpg"))) + [
    os.path.join(ROOT, "backend/frs/samples/480.jpg")
]
OUTS = [
    os.path.join(ROOT, "frontend/frs_react/tests/fixtures/reference.json"),
    os.path.join(ROOT, "mobile/face_app/test/fixtures/reference.json"),
]


def b64f32(v: np.ndarray) -> str:
    return base64.b64encode(np.asarray(v, dtype="<f4").tobytes()).decode()


def main():
    rng = np.random.default_rng(42)
    pipe = FacePipeline(os.path.join(ROOT, "models"))

    similarity_cases = []
    for _ in range(20):
        src = rng.uniform(0, 400, size=(5, 2))
        dst = rng.uniform(0, 112, size=(5, 2)) if rng.random() < 0.5 else (
            # a near-similarity mapping with noise
            (src @ np.array([[0.4, 0.1], [-0.1, 0.4]]).T) + rng.normal(0, 2, (5, 2)) + 10
        )
        similarity_cases.append({"src": src.tolist(), "dst": dst.tolist(), "m": umeyama(src, dst).ravel().tolist()})

    images = []
    for path in IMAGES:
        img = cv2.imread(path)
        faces = pipe.detector.detect(img)
        rows = []
        for i, f in enumerate(faces):
            aligned = align_face(img, f.kps)
            emb = pipe.embedder.embed_aligned(aligned)
            row = {
                "bbox": [float(x) for x in f.bbox],
                "kps": f.kps.tolist(),
                "score": f.score,
                "embedding": b64f32(emb),
            }
            if i == 0 and len(images) < 2:
                row["alignedRgb"] = base64.b64encode(aligned.tobytes()).decode()
            rows.append(row)
        images.append({"path": os.path.relpath(path, ROOT).replace("\\", "/"), "width": img.shape[1], "height": img.shape[0], "faces": rows})

    data = {
        "model": "w600k_mbf.onnx",
        "detector": "det_500m.onnx",
        "template": ARCFACE_TEMPLATE.tolist(),
        "similarityCases": similarity_cases,
        "images": images,
        "base64Probe": {"values": [0.0, 1.0, -2.5, 3.14159], "b64": b64f32(np.array([0.0, 1.0, -2.5, 3.14159]))},
    }
    for out in OUTS:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(data, fh)
        print("wrote", out, f"({sum(len(i['faces']) for i in images)} faces)")


if __name__ == "__main__":
    main()
