import 'dart:typed_data';

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

import '../core/embedding.dart';
import '../core/preprocess.dart';

/// ArcFace style embedding model on ONNX Runtime Mobile. Uses the very same w600k_mbf.onnx file as
/// the web app, so phones and browsers produce embeddings in one vector space.
class FaceEmbedder {
  FaceEmbedder._(this._session, this.inputSize);

  final OrtSession _session;
  final int inputSize;
  bool _batchSupported = true;

  static Future<FaceEmbedder> load({
    required String asset,
    int inputSize = 112,
    List<OrtProvider> providers = const [OrtProvider.XNNPACK, OrtProvider.CPU],
  }) async {
    final session = await OnnxRuntime().createSessionFromAsset(
      asset,
      options: OrtSessionOptions(providers: providers, intraOpNumThreads: 2),
    );
    final embedder = FaceEmbedder._(session, inputSize);
    await embedder.embed(Uint8List(inputSize * inputSize * 3)); // warm up
    return embedder;
  }

  /// [rgb]: aligned inputSize x inputSize interleaved RGB. Returns an L2 normalised embedding.
  Future<Float32List> embed(Uint8List rgb, {bool flipTta = true}) async {
    final s = inputSize, plane = 3 * s * s;
    if (!flipTta) return l2Normalize(await _run(rgbToTensor(rgb, s, s), 1));

    if (_batchSupported) {
      try {
        final data = Float32List(2 * plane);
        rgbToTensor(rgb, s, s, out: data);
        rgbToTensor(rgb, s, s, out: data, offset: plane, flip: true);
        return _sumHalves(await _run(data, 2));
      } catch (_) {
        _batchSupported = false; // fall back to two single runs
      }
    }
    final a = await _run(rgbToTensor(rgb, s, s), 1);
    final b = await _run(rgbToTensor(rgb, s, s, flip: true), 1);
    return _sumHalves([...a, ...b]);
  }

  Future<List<double>> _run(Float32List data, int batch) async {
    final input = await OrtValue.fromList(data, [batch, 3, inputSize, inputSize]);
    try {
      final outputs = await _session.run({_session.inputNames.first: input});
      final out = outputs[_session.outputNames.first]!;
      final values = (await out.asFlattenedList()).cast<num>().map((v) => v.toDouble()).toList();
      for (final o in outputs.values) {
        await o.dispose();
      }
      return values;
    } finally {
      await input.dispose();
    }
  }

  /// InsightFace flip TTA: add the features of the face and its mirror image, then normalise.
  static Float32List _sumHalves(List<double> batched) {
    final dim = batched.length ~/ 2;
    return l2Normalize([for (var i = 0; i < dim; i++) batched[i] + batched[i + dim]]);
  }

  Future<void> close() => _session.close();
}
