import 'geometry.dart';

const int alignedFaceSize = 112;

/// ArcFace / InsightFace 5-point template for a 112x112 crop. Order in image coordinates of the
/// upright, non-mirrored frame: image-left eye, image-right eye, nose tip, image-left and
/// image-right mouth corner.
const List<Point2> arcfaceTemplate112 = [
  Point2(38.2946, 51.6963),
  Point2(73.5318, 51.5014),
  Point2(56.0252, 71.7366),
  Point2(41.5493, 92.3655),
  Point2(70.7299, 92.2041),
];

/// Matrix mapping source pixels to the aligned crop. [landmarks]: 5 points in template order, or
/// 4 points (eyes, nose, mouth centre).
Affine alignmentMatrix(List<Point2> landmarks, {int size = alignedFaceSize}) {
  final List<Point2> template;
  if (landmarks.length == 5) {
    template = arcfaceTemplate112;
  } else if (landmarks.length == 4) {
    final m = Point2((arcfaceTemplate112[3].x + arcfaceTemplate112[4].x) / 2,
        (arcfaceTemplate112[3].y + arcfaceTemplate112[4].y) / 2);
    template = [...arcfaceTemplate112.take(3), m];
  } else {
    throw ArgumentError('expected 5 or 4 landmarks, got ${landmarks.length}');
  }
  final s = size / alignedFaceSize;
  return estimateSimilarityTransform(landmarks, [for (final p in template) Point2(p.x * s, p.y * s)]);
}

/// Orders two points left to right in the image (robust to "left"/"right" naming conventions).
(Point2, Point2) leftRight(Point2 a, Point2 b) => a.x <= b.x ? (a, b) : (b, a);
