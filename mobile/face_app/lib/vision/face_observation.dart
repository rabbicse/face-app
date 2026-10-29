import 'dart:ui' show Rect;

import '../core/geometry.dart';
import '../core/pose.dart';

/// One detected face in *upright* frame coordinates.
class FaceObservation {
  const FaceObservation({
    required this.points5,
    required this.box,
    required this.pose,
    this.eyeOpenProbability,
    this.trackingId,
  });

  /// image-left eye, image-right eye, nose, image-left / image-right mouth corner
  final List<Point2> points5;
  final Rect box;
  final HeadPose pose;
  final double? eyeOpenProbability;
  final int? trackingId;

  double get interOcular => points5[0].distanceTo(points5[1]);
}
