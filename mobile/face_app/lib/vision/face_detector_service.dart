import 'dart:io' show Platform;
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Size;

import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import '../core/alignment.dart';
import '../core/geometry.dart';
import '../core/pose.dart';
import 'camera_frame.dart';
import 'face_observation.dart';

/// Google ML Kit face detector (on-device): 5 landmarks, head Euler angles, eye open probability.
class FaceDetectorService {
  /// [accurate] is needed for a reliable yaw (registration); identification uses the fast mode.
  FaceDetectorService({required bool accurate})
      : _detector = FaceDetector(
          options: FaceDetectorOptions(
            performanceMode: accurate ? FaceDetectorMode.accurate : FaceDetectorMode.fast,
            enableLandmarks: true,
            enableClassification: true,
            enableTracking: true,
            minFaceSize: 0.2,
          ),
        );

  final FaceDetector _detector;

  Future<FaceObservation?> detect(CameraFrame frame) async {
    final faces = await _detector.processImage(_toInputImage(frame));
    if (faces.isEmpty) return null;
    final face = faces.reduce((a, b) =>
        a.boundingBox.width * a.boundingBox.height >= b.boundingBox.width * b.boundingBox.height ? a : b);
    return toObservation(face);
  }

  static FaceObservation? toObservation(Face face) {
    Point2? at(FaceLandmarkType t) {
      final p = face.landmarks[t]?.position;
      return p == null ? null : Point2(p.x.toDouble(), p.y.toDouble());
    }

    final le = at(FaceLandmarkType.leftEye), re = at(FaceLandmarkType.rightEye);
    final nose = at(FaceLandmarkType.noseBase);
    final lm = at(FaceLandmarkType.leftMouth), rm = at(FaceLandmarkType.rightMouth);
    if (le == null || re == null || nose == null || lm == null || rm == null) return null;

    final (eyeL, eyeR) = leftRight(le, re);
    final (mouthL, mouthR) = leftRight(lm, rm);
    final l = face.leftEyeOpenProbability, r = face.rightEyeOpenProbability;
    return FaceObservation(
      points5: [eyeL, eyeR, nose, mouthL, mouthR],
      box: face.boundingBox,
      pose: HeadPose(face.headEulerAngleY ?? 0, face.headEulerAngleX ?? 0, face.headEulerAngleZ ?? 0),
      eyeOpenProbability: l != null && r != null ? math.min(l, r) : null,
      trackingId: face.trackingId,
    );
  }

  static InputImage _toInputImage(CameraFrame frame) {
    final Uint8List bytes;
    if (frame.planes.length == 1) {
      bytes = frame.planes.first.bytes;
    } else {
      final b = BytesBuilder(copy: false);
      for (final p in frame.planes) {
        b.add(p.bytes);
      }
      bytes = b.toBytes();
    }
    final format = switch (frame.layout) {
      PixelLayout.bgra8888 => InputImageFormat.bgra8888,
      PixelLayout.nv21 => InputImageFormat.nv21,
      PixelLayout.yuv420 => Platform.isIOS ? InputImageFormat.yuv420 : InputImageFormat.yuv_420_888,
    };
    return InputImage.fromBytes(
      bytes: bytes,
      metadata: InputImageMetadata(
        size: Size(frame.width.toDouble(), frame.height.toDouble()),
        rotation: InputImageRotationValue.fromRawValue(frame.rotation) ?? InputImageRotation.rotation0deg,
        format: format,
        bytesPerRow: frame.planes.first.bytesPerRow,
      ),
    );
  }

  Future<void> close() => _detector.close();
}
