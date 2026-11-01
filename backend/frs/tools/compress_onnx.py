"""
Shrinks an ArcFace ONNX model for the clients by storing the weights in a smaller type; activations and
arithmetic stay float32, so the clients need no changes.

* ``--mode fp16`` (default): weights stored as float16 + a Cast node. ONNX Runtime folds the Cast when the
  session is created, so speed is unchanged and accuracy is identical in practice. w600k_mbf: 13.6 -> 6.9 MB,
  cosine to the fp32 embedding >= 0.99999.
* ``--mode int8``: per-output-channel int8 + DequantizeLinear. 13.6 -> 3.5 MB, but ONNX Runtime does *not*
  fold the dequantisation (measured ~65% slower on CPU) and similarities move by up to ~0.01.

    python tools/compress_onnx.py --input w600k_mbf.onnx --output w600k_mbf.fp16.onnx --verify-images "samples/*.jp*g"

--verify-images embeds every face YuNet finds in the images with both models and reports the per-face cosine
similarity between fp32 and compressed embeddings and the largest change of any face-pair similarity.
"""
import argparse
import glob
import os
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))

MIN_OPSET = 13  # per-axis DequantizeLinear


def _weight_names(model: onnx.ModelProto, min_elements: int) -> list:
    inits = {i.name: i for i in model.graph.initializer}
    names = {n.input[1] for n in model.graph.node if n.op_type in ('Conv', 'Gemm') and len(n.input) > 1}
    return sorted(n for n in names if n in inits and int(np.prod(inits[n].dims)) >= min_elements)


def _prepend(graph, nodes) -> None:
    """Puts nodes first, which keeps the graph topologically sorted."""
    all_nodes = list(nodes) + list(graph.node)
    del graph.node[:]
    graph.node.extend(all_nodes)


def compress_fp16(model: onnx.ModelProto, min_elements: int = 1024) -> onnx.ModelProto:
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    casts = []
    for name in _weight_names(model, min_elements):
        w = numpy_helper.to_array(inits[name])
        if w.dtype != np.float32:
            continue
        if np.abs(w).max() > 65000:
            raise ValueError(f'{name} does not fit into float16')
        graph.initializer.remove(inits[name])
        graph.initializer.append(numpy_helper.from_array(w.astype(np.float16), f'{name}_fp16'))
        casts.append(helper.make_node('Cast', [f'{name}_fp16'], [name], name=f'{name}_cast', to=onnx.TensorProto.FLOAT))
    _prepend(graph, casts)
    onnx.checker.check_model(model)
    return model


def compress(model: onnx.ModelProto, min_elements: int = 1024) -> onnx.ModelProto:
    """int8 weight-only variant (see module docstring)."""
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    for node in graph.node:
        if node.op_type == 'Gemm' and any(a.name == 'transB' and a.i == 0 for a in node.attribute):
            raise ValueError('Gemm without transB is not supported (output channels must be axis 0)')

    new_nodes = []
    for name in _weight_names(model, min_elements):
        init = inits[name]
        w = numpy_helper.to_array(init).astype(np.float32)
        flat = np.abs(w.reshape(w.shape[0], -1)).max(axis=1)
        scale = np.where(flat > 0, flat / 127.0, 1.0).astype(np.float32)
        q = np.clip(np.round(w / scale.reshape(-1, *([1] * (w.ndim - 1)))), -127, 127).astype(np.int8)

        graph.initializer.remove(init)
        graph.initializer.extend([
            numpy_helper.from_array(q, f'{name}_q'),
            numpy_helper.from_array(scale, f'{name}_scale'),
            numpy_helper.from_array(np.zeros(w.shape[0], np.int8), f'{name}_zp'),
        ])
        new_nodes.append(helper.make_node('DequantizeLinear', [f'{name}_q', f'{name}_scale', f'{name}_zp'], [name],
                                          name=f'{name}_dequantize', axis=0))

    _prepend(graph, new_nodes)
    for opset in model.opset_import:
        if opset.domain in ('', 'ai.onnx') and opset.version < MIN_OPSET:
            opset.version = MIN_OPSET
    model.ir_version = max(model.ir_version, 7)
    onnx.checker.check_model(model)
    return model


def verify(fp32_path: str, int8_path: str, images: list) -> None:
    import cv2
    import onnxruntime as ort
    from face_pipeline.alignment import warp_face
    from face_pipeline.preprocess import to_model_input
    from client_embedding_demo import DEFAULT_DETECTOR, FaceEmbedder

    FaceEmbedder(fp32_path, DEFAULT_DETECTOR)  # downloads YuNet if needed
    detector = cv2.FaceDetectorYN.create(DEFAULT_DETECTOR, '', (320, 320), score_threshold=0.7)
    crops = []
    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue
        detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = detector.detect(img)
        for f in faces if faces is not None else []:
            pts = f[4:14].reshape(5, 2)
            eyes, mouth = sorted(pts[:2].tolist()), sorted(pts[3:5].tolist())
            crops.append(cv2.cvtColor(warp_face(img, [eyes[0], eyes[1], pts[2].tolist(), mouth[0], mouth[1]]),
                                      cv2.COLOR_BGR2RGB))
    if not crops:
        raise SystemExit('no faces found in the verification images')

    options = ort.SessionOptions()
    options.log_severity_level = 3

    def embed(path):
        s = ort.InferenceSession(path, options, providers=['CPUExecutionProvider'])
        x = to_model_input(np.stack(crops))
        out = np.concatenate([s.run(None, {s.get_inputs()[0].name: x[i:i + 32]})[0] for i in range(0, len(x), 32)])
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    a, b = embed(fp32_path), embed(int8_path)
    per_face = (a * b).sum(axis=1)
    pair_delta = np.abs(a @ a.T - b @ b.T).max()
    print(f'{len(crops)} faces: cosine(fp32, compressed) min {per_face.min():.5f}, mean {per_face.mean():.5f}; '
          f'max change of a face-pair similarity {pair_delta:.5f}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--mode', choices=['fp16', 'int8'], default='fp16')
    p.add_argument('--verify-images', nargs='*', default=[])
    args = p.parse_args()

    model = (compress_fp16 if args.mode == 'fp16' else compress)(onnx.load(args.input))
    onnx.save(model, args.output)
    print(f'{args.input}: {os.path.getsize(args.input) / 1e6:.1f} MB -> {args.output}: '
          f'{os.path.getsize(args.output) / 1e6:.1f} MB')
    images = [f for pattern in args.verify_images for f in glob.glob(pattern)]
    if images:
        verify(args.input, args.output, images)


if __name__ == '__main__':
    main()
