"""
Python reference client for the client-side embedding API.

It computes the 512-d embedding exactly like the web and Flutter apps (5 landmarks -> ArcFace
alignment -> MobileFaceNet ONNX with flip TTA -> L2 normalisation) and talks to
/api/v1/embeddings/*. Useful to enrol from photos, to batch test thresholds and to check that the
devices produce compatible vectors.

Requirements: pip install numpy opencv-python-headless onnxruntime requests

    # models (w600k_mbf.onnx is set up by `npm run setup:face` in frontend/frs_react)
    python examples/client_embedding_demo.py embed     --image samples/480.jpg
    python examples/client_embedding_demo.py compare   --image a.jpg --image2 b.jpg
    python examples/client_embedding_demo.py register  --image samples/480.jpg --name "Mehedi" --email m@example.com
    python examples/client_embedding_demo.py identify  --image samples/480.jpg
    python examples/client_embedding_demo.py benchmark --image samples/480.jpg
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import zlib

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from face_pipeline.alignment import warp_face  # noqa: E402
from face_pipeline.embedding import cosine_similarity, encode_base64, l2_normalize  # noqa: E402
from face_pipeline.preprocess import to_model_input  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, '..', '..', '..', 'frontend', 'frs_react', 'public', 'models', 'arcface',
                             'w600k_mbf.onnx')
DEFAULT_DETECTOR = os.path.join(HERE, '..', 'models', 'face_detection_yunet_2023mar.onnx')
YUNET_URL = ('https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/'
             'face_detection_yunet_2023mar.onnx')
MODEL_ID = 'w600k_mbf'


class FaceEmbedder:
    def __init__(self, model_path: str, detector_path: str, flip_tta: bool = True):
        import onnxruntime as ort

        if not os.path.exists(detector_path):
            print(f'downloading YuNet face detector -> {detector_path}')
            os.makedirs(os.path.dirname(detector_path), exist_ok=True)
            urllib.request.urlretrieve(YUNET_URL, detector_path)
        self.detector = cv2.FaceDetectorYN.create(detector_path, '', (320, 320), score_threshold=0.7)
        options = ort.SessionOptions()
        options.log_severity_level = 3  # the model declares batch 1 but accepts 2 (flip TTA)
        self.session = ort.InferenceSession(model_path, options, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.flip_tta = flip_tta

    def landmarks(self, bgr: np.ndarray) -> np.ndarray:
        """5 points in ArcFace order (image-left eye, image-right eye, nose, mouth left, mouth right)."""
        h, w = bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(bgr)
        if faces is None or len(faces) == 0:
            raise ValueError('no face found')
        face = max(faces, key=lambda f: f[2] * f[3])  # largest face
        pts = face[4:14].reshape(5, 2)
        eyes = sorted(pts[0:2].tolist())       # order left to right in the image, like the clients do
        mouth = sorted(pts[3:5].tolist())
        return np.array([eyes[0], eyes[1], pts[2].tolist(), mouth[0], mouth[1]], dtype=np.float64)

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(warp_face(bgr, self.landmarks(bgr)), cv2.COLOR_BGR2RGB)
        batch = np.stack([rgb, rgb[:, ::-1]]) if self.flip_tta else rgb[None]
        feats = self.session.run(None, {self.input_name: to_model_input(batch)})[0]
        return l2_normalize(feats.sum(axis=0))


def person_id_from_email(email: str) -> int:
    """Same as frontend/frs_react/src/face/react/personId.ts (FNV-1a, 31 bit)."""
    h = 0x811c9dc5
    for ch in email.strip().lower():
        h = ((h ^ ord(ch)) * 0x01000193) & 0xFFFFFFFF
    return (h >> 1) or 1


def post(api: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(f'{api}{path}', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req) as res:
        return json.loads(res.read())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['embed', 'compare', 'register', 'identify', 'benchmark'])
    p.add_argument('--image', required=True)
    p.add_argument('--image2')
    p.add_argument('--name')
    p.add_argument('--email', default='')
    p.add_argument('--api', default=os.getenv('FRS_API', 'http://localhost:5000'))
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--detector', default=DEFAULT_DETECTOR)
    p.add_argument('--no-flip', action='store_true', help='disable flip test time augmentation')
    args = p.parse_args()

    embedder = FaceEmbedder(args.model, args.detector, flip_tta=not args.no_flip)
    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f'cannot read {args.image}')

    if args.command == 'benchmark':
        embedder.embed(image)
        runs = 50
        t0 = time.perf_counter()
        for _ in range(runs):
            pts = embedder.landmarks(image)
        t1 = time.perf_counter()
        for _ in range(runs):
            warp_face(image, pts)
        t2 = time.perf_counter()
        for _ in range(runs):
            embedder.embed(image)
        t3 = time.perf_counter()
        print(f'detection+landmarks: {(t1 - t0) / runs * 1000:.1f} ms, alignment: {(t2 - t1) / runs * 1000:.2f} ms, '
              f'full pipeline: {(t3 - t2) / runs * 1000:.1f} ms (flip TTA: {not args.no_flip})')
        return

    emb = embedder.embed(image)
    if args.command == 'embed':
        print(json.dumps({'model': MODEL_ID, 'dim': int(emb.shape[0]), 'norm': float(np.linalg.norm(emb)),
                          'embedding_base64': encode_base64(emb), 'head': [round(float(v), 5) for v in emb[:8]]},
                         indent=1))
    elif args.command == 'compare':
        if not args.image2:
            sys.exit('--image2 is required')
        other = embedder.embed(cv2.imread(args.image2))
        score = cosine_similarity(emb, other)
        print(f'cosine similarity: {score:.4f} -> {"same person" if score >= 0.45 else "different people"}')
    elif args.command == 'register':
        if not args.name:
            sys.exit('--name is required')
        person = {'person_id': person_id_from_email(args.email or args.name), 'name': args.name, 'email': args.email}
        res = post(args.api, '/api/v1/embeddings/register', {
            'model': MODEL_ID, 'person': person,
            'samples': [{'embedding': encode_base64(emb), 'pose_label': 'frontal'}],
        })
        print(json.dumps(res, indent=1))
    elif args.command == 'identify':
        print(json.dumps(post(args.api, '/api/v1/embeddings/identify',
                              {'model': MODEL_ID, 'embedding': encode_base64(emb)}), indent=1))


if __name__ == '__main__':
    main()
