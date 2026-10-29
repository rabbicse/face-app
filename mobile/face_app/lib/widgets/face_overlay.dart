import 'package:flutter/material.dart';

import '../vision/face_observation.dart';

/// Draws the face box and alignment points. Coordinates come from the upright frame and are scaled
/// to the preview; front camera previews are mirrored, so x is flipped as well.
class FaceOverlayPainter extends CustomPainter {
  FaceOverlayPainter({
    required this.face,
    required this.frameSize,
    required this.mirror,
    required this.color,
    this.progress,
  });

  final FaceObservation? face;
  final Size frameSize;
  final bool mirror;
  final Color color;
  final double? progress;

  @override
  void paint(Canvas canvas, Size size) {
    final guide = Rect.fromCenter(
        center: size.center(Offset.zero), width: size.width * 0.65, height: size.width * 0.65 * 1.3);
    canvas.drawOval(guide, Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4
      ..color = Colors.white24);
    if (progress != null && progress! > 0) {
      canvas.drawArc(guide, -1.5708, 6.2832 * progress!.clamp(0, 1), false, Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 4
        ..color = Colors.lightBlueAccent);
    }

    final f = face;
    if (f == null || frameSize.isEmpty) return;
    final sx = size.width / frameSize.width, sy = size.height / frameSize.height;
    Offset map(double x, double y) => Offset(mirror ? size.width - x * sx : x * sx, y * sy);

    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = color;
    canvas.drawRect(Rect.fromPoints(map(f.box.left, f.box.top), map(f.box.right, f.box.bottom)), paint);
    final dot = Paint()..color = color;
    for (final p in f.points5) {
      canvas.drawCircle(map(p.x, p.y), 3, dot);
    }
  }

  @override
  bool shouldRepaint(FaceOverlayPainter old) =>
      old.face != face || old.color != color || old.progress != progress || old.frameSize != frameSize;
}
