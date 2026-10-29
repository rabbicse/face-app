import 'dart:typed_data';

import '../config.dart';
import '../core/alignment.dart';
import '../core/preprocess.dart';
import '../core/quality.dart';
import 'camera_frame.dart';
import 'embedder.dart';
import 'face_detector_service.dart';
import 'face_observation.dart';

class FrameAnalysis {
  const FrameAnalysis({
    required this.face,
    required this.crop,
    required this.quality,
    required this.sharpness,
    required this.brightness,
  });

  final FaceObservation face;

  /// aligned 112x112 RGB crop
  final Uint8List crop;
  final QualityReport quality;
  final double sharpness;
  final double brightness;
}

/// Everything on the device: ML Kit landmarks + pose -> aligned crop sampled from the raw frame ->
/// quality gate -> ArcFace embedding (ONNX Runtime). Only the 512-d vector is sent to the server.
class FaceEngine {
  FaceEngine._(this._detector, this._embedder);

  final FaceDetectorService _detector;
  final FaceEmbedder _embedder;

  static Future<FaceEngine> create({required bool accuratePose}) async {
    final embedder = await FaceEmbedder.load(
        asset: AppConfig.embeddingModelAsset, inputSize: AppConfig.embeddingInputSize);
    return FaceEngine._(FaceDetectorService(accurate: accuratePose), embedder);
  }

  Future<FrameAnalysis?> analyze(CameraFrame frame, {QualityThresholds thresholds = const QualityThresholds()}) async {
    final face = await _detector.detect(frame);
    if (face == null) return null;
    return alignAndAssess(frame, face, thresholds: thresholds);
  }

  static FrameAnalysis alignAndAssess(CameraFrame frame, FaceObservation face,
      {QualityThresholds thresholds = const QualityThresholds()}) {
    const size = AppConfig.embeddingInputSize;
    final crop = warpFaceFromFrame(frame, alignmentMatrix(face.points5, size: size), size);
    final gray = rgbToGray(crop, size, size);
    final sharpness = laplacianVariance(gray, size, size);
    final brightness = meanValue(gray);
    final quality = assessQuality(
      box: face.box,
      frameWidth: frame.uprightWidth.toDouble(),
      frameHeight: frame.uprightHeight.toDouble(),
      interOcular: face.interOcular,
      sharpness: sharpness,
      brightness: brightness,
      eyeOpenProbability: face.eyeOpenProbability,
      t: thresholds,
    );
    return FrameAnalysis(face: face, crop: crop, quality: quality, sharpness: sharpness, brightness: brightness);
  }

  Future<Float32List> embed(FrameAnalysis analysis) => _embedder.embed(analysis.crop, flipTta: AppConfig.flipTta);

  Future<void> close() async {
    await _detector.close();
    await _embedder.close();
  }
}
