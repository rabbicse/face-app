import 'dart:math' as math;
import 'dart:ui' show Rect;

class QualityThresholds {
  const QualityThresholds({
    this.minInterOcular = 32,
    this.minSharpness = 40,
    this.minBrightness = 50,
    this.maxBrightness = 215,
    this.maxOutside = 0.05,
    this.minEyeOpenProbability = 0.4,
  });

  /// eye distance in frame pixels; the 112px template has ~35px, below that the crop is upscaled
  final double minInterOcular;
  final double minSharpness;
  final double minBrightness;
  final double maxBrightness;

  /// fraction of the face box that may lie outside the frame
  final double maxOutside;

  /// ML Kit classification; ignored when not available
  final double minEyeOpenProbability;
}

enum QualityIssue { tooFar, blurry, tooDark, tooBright, outOfFrame, eyesClosed }

const Map<QualityIssue, String> qualityMessages = {
  QualityIssue.tooFar: 'Move closer to the camera',
  QualityIssue.blurry: 'Hold still',
  QualityIssue.tooDark: 'Find better lighting',
  QualityIssue.tooBright: 'Too bright, avoid direct light',
  QualityIssue.outOfFrame: 'Center your face in the frame',
  QualityIssue.eyesClosed: 'Keep your eyes open',
};

class QualityReport {
  const QualityReport(this.issues, this.score);
  final List<QualityIssue> issues;

  /// 0..1, higher is better
  final double score;

  bool get ok => issues.isEmpty;
}

QualityReport assessQuality({
  required Rect box,
  required double frameWidth,
  required double frameHeight,
  required double interOcular,
  required double sharpness,
  required double brightness,
  double? eyeOpenProbability,
  QualityThresholds t = const QualityThresholds(),
}) {
  final issues = <QualityIssue>[];
  final visibleW = math.min(box.right, frameWidth) - math.max(box.left, 0);
  final visibleH = math.min(box.bottom, frameHeight) - math.max(box.top, 0);
  final visible = math.max(0.0, visibleW) * math.max(0.0, visibleH) / math.max(1e-6, box.width * box.height);

  if (1 - visible > t.maxOutside) issues.add(QualityIssue.outOfFrame);
  if (interOcular < t.minInterOcular) issues.add(QualityIssue.tooFar);
  if (sharpness < t.minSharpness) issues.add(QualityIssue.blurry);
  if (brightness < t.minBrightness) issues.add(QualityIssue.tooDark);
  if (brightness > t.maxBrightness) issues.add(QualityIssue.tooBright);
  if (eyeOpenProbability != null && eyeOpenProbability < t.minEyeOpenProbability) {
    issues.add(QualityIssue.eyesClosed);
  }

  double clamp(double v) => v.clamp(0.0, 1.0);
  final size = clamp(interOcular / (2 * t.minInterOcular));
  final sharp = clamp(sharpness / (4 * t.minSharpness));
  final mid = (t.minBrightness + t.maxBrightness) / 2;
  final light = clamp(1 - (brightness - mid).abs() / (mid - t.minBrightness + 1e-6) * 0.5);
  final score = (clamp(visible) * (0.4 * size + 0.4 * sharp + 0.2 * light) * 1000).round() / 1000;
  return QualityReport(issues, score);
}
