"""
Export an ArcFace backbone of this repo (src/recognition/arc_face/backbones) to ONNX for the web and
Flutter clients (dynamic batch, input 'input.1' [N, 3, 112, 112] RGB in [-1, 1], output [N, 512]).

    python tools/export_onnx.py --network mbf --weights ../models/backbone_mbf.pth --output w600k_custom.onnx
    python tools/export_onnx.py --network r50 --weights ../models/backbone_r50.pth --output r50.onnx --fp16

Small backbones (mbf ~13 MB, r18/r34) are the realistic choice on phones and in browsers; r100 is ~250 MB.
After exporting, put the file into frontend/frs_react/public/models/arcface/ and
mobile/face_app/assets/models/, and register a *new* model id with the backend (FRS_EMBEDDING_MODELS):
embeddings of different models are never comparable, so everyone has to re-register.
"""
import argparse
import hashlib
import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def load_backbone(network: str, weights: str | None):
    import torch
    from recognition.arc_face.backbones import get_model

    model = get_model(network, dropout=0, fp16=False)
    if weights:
        state = torch.load(weights, map_location='cpu', weights_only=True)
        state = {k.removeprefix('module.'): v for k, v in state.items()}  # DataParallel checkpoints
        model.load_state_dict(state)
    else:
        print('warning: no --weights given, exporting random weights (only useful to test the tooling)')
    return model.eval()


def export(model, output: str, opset: int) -> None:
    import torch

    dummy = torch.randn(1, 3, 112, 112)
    kwargs = dict(input_names=['input.1'], output_names=['embedding'], opset_version=opset,
                  do_constant_folding=True,
                  dynamic_axes={'input.1': {0: 'batch'}, 'embedding': {0: 'batch'}})
    if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
        kwargs['dynamo'] = False  # the TorchScript exporter keeps the graph simple for onnxruntime-web
    with torch.no_grad():
        torch.onnx.export(model, dummy, output, **kwargs)


def to_fp16(path: str) -> None:
    import onnx
    from onnxconverter_common import float16

    model = float16.convert_float_to_float16(onnx.load(path), keep_io_types=True)
    onnx.save(model, path)


def verify(model, output: str, tolerance: float) -> float:
    import onnxruntime as ort
    import torch

    x = np.random.default_rng(0).uniform(-1, 1, (2, 3, 112, 112)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    session = ort.InferenceSession(output, providers=['CPUExecutionProvider'])
    actual = session.run(None, {session.get_inputs()[0].name: x})[0]
    a = expected / np.linalg.norm(expected, axis=1, keepdims=True)
    b = actual / np.linalg.norm(actual, axis=1, keepdims=True)
    cosine = float((a * b).sum(axis=1).min())
    if cosine < 1 - tolerance:
        raise SystemExit(f'ONNX output differs from PyTorch: min cosine {cosine:.6f}')
    return cosine


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--network', default='mbf', help='mbf, r18, r34, r50, r100, r200')
    p.add_argument('--weights', help='PyTorch state_dict (.pth)')
    p.add_argument('--output', required=True)
    p.add_argument('--opset', type=int, default=17)
    p.add_argument('--fp16', action='store_true', help='halve the file size (needs onnxconverter-common)')
    args = p.parse_args()

    model = load_backbone(args.network, args.weights)
    export(model, args.output, args.opset)
    if args.fp16:
        to_fp16(args.output)
    cosine = verify(model, args.output, tolerance=5e-3 if args.fp16 else 1e-5)

    data = open(args.output, 'rb').read()
    print(f'exported {args.output}: {len(data) / 1e6:.1f} MB, sha256 {hashlib.sha256(data).hexdigest()}')
    print(f'onnxruntime vs pytorch: min cosine {cosine:.6f} (dynamic batch verified with N=2)')


if __name__ == '__main__':
    main()
