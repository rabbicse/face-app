import 'dart:math' as math;

class Point2 {
  const Point2(this.x, this.y);
  final double x;
  final double y;

  double distanceTo(Point2 o) => math.sqrt((x - o.x) * (x - o.x) + (y - o.y) * (y - o.y));

  @override
  String toString() => 'Point2($x, $y)';
}

/// Row major 2x3 affine matrix: x' = a*x + b*y + c, y' = d*x + e*y + f.
class Affine {
  const Affine(this.a, this.b, this.c, this.d, this.e, this.f);
  final double a, b, c, d, e, f;

  List<double> get values => [a, b, c, d, e, f];

  Point2 apply(Point2 p) => Point2(a * p.x + b * p.y + c, d * p.x + e * p.y + f);

  Affine inverse() {
    final det = a * e - b * d;
    if (det.abs() < 1e-12) throw StateError('affine matrix is not invertible');
    final ia = e / det, ib = -b / det, id = -d / det, ie = a / det;
    return Affine(ia, ib, -(ia * c + ib * f), id, ie, -(id * c + ie * f));
  }
}

/// Least squares non-reflective similarity transform mapping [src] onto [dst]
/// (closed form of Umeyama for 2D, identical to skimage's SimilarityTransform).
Affine estimateSimilarityTransform(List<Point2> src, List<Point2> dst) {
  if (src.length != dst.length || src.length < 2) {
    throw ArgumentError('expected two point sets of equal length >= 2');
  }
  final n = src.length;
  var sx = 0.0, sy = 0.0, dx = 0.0, dy = 0.0;
  for (var i = 0; i < n; i++) {
    sx += src[i].x;
    sy += src[i].y;
    dx += dst[i].x;
    dy += dst[i].y;
  }
  sx /= n;
  sy /= n;
  dx /= n;
  dy /= n;

  var denom = 0.0, aNum = 0.0, bNum = 0.0;
  for (var i = 0; i < n; i++) {
    final px = src[i].x - sx, py = src[i].y - sy;
    final qx = dst[i].x - dx, qy = dst[i].y - dy;
    denom += px * px + py * py;
    aNum += px * qx + py * qy;
    bNum += px * qy - py * qx;
  }
  if (denom < 1e-12) throw ArgumentError('source points are degenerate');
  final a = aNum / denom, b = bNum / denom;
  return Affine(a, -b, dx - (a * sx - b * sy), b, a, dy - (b * sx + a * sy));
}
