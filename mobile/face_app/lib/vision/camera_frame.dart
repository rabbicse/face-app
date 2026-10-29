import 'dart:typed_data';

import '../core/geometry.dart';

enum PixelLayout {
  /// Android: one plane, Y (rowStride) followed by interleaved V/U at half resolution
  nv21,

  /// three planes Y, U, V with their own row / pixel strides (Android YUV_420_888)
  yuv420,

  /// iOS: one plane, 4 bytes per pixel B, G, R, A
  bgra8888,
}

class FramePlane {
  const FramePlane(this.bytes, this.bytesPerRow, [this.bytesPerPixel = 1]);
  final Uint8List bytes;
  final int bytesPerRow;
  final int bytesPerPixel;
}

/// A raw camera frame (independent of the camera plugin, so the pixel code is unit testable).
///
/// [rotation] is the clockwise rotation in degrees that makes the raw buffer upright - the value
/// given to ML Kit, whose landmarks are therefore in *upright* coordinates.
class CameraFrame {
  const CameraFrame({
    required this.width,
    required this.height,
    required this.layout,
    required this.planes,
    this.rotation = 0,
  });

  final int width;
  final int height;
  final PixelLayout layout;
  final List<FramePlane> planes;
  final int rotation;

  int get uprightWidth => rotation % 180 == 0 ? width : height;
  int get uprightHeight => rotation % 180 == 0 ? height : width;

  /// Maps upright coordinates to raw buffer coordinates (pixel centres on integers).
  (double, double) toRaw(double xu, double yu) {
    switch (rotation) {
      case 90:
        return (yu, height - 1 - xu);
      case 180:
        return (width - 1 - xu, height - 1 - yu);
      case 270:
        return (width - 1 - yu, xu);
      default:
        return (xu, yu);
    }
  }

  /// RGB of raw pixel (x, y), clamped to the frame; writes 3 values into [out] at [o].
  void _rgbAt(int x, int y, List<int> out, int o) {
    x = x < 0 ? 0 : (x >= width ? width - 1 : x);
    y = y < 0 ? 0 : (y >= height ? height - 1 : y);
    switch (layout) {
      case PixelLayout.bgra8888:
        final p = planes[0];
        final i = y * p.bytesPerRow + x * 4;
        out[o] = p.bytes[i + 2];
        out[o + 1] = p.bytes[i + 1];
        out[o + 2] = p.bytes[i];
        return;
      case PixelLayout.nv21:
        final p = planes[0];
        final yv = p.bytes[y * p.bytesPerRow + x];
        final uvIndex = height * p.bytesPerRow + (y >> 1) * p.bytesPerRow + (x & ~1);
        _yuvToRgb(yv, p.bytes[uvIndex + 1], p.bytes[uvIndex], out, o);
        return;
      case PixelLayout.yuv420:
        final yp = planes[0], up = planes[1], vp = planes[2];
        final yv = yp.bytes[y * yp.bytesPerRow + x];
        final cx = x >> 1, cy = y >> 1;
        final u = up.bytes[cy * up.bytesPerRow + cx * up.bytesPerPixel];
        final v = vp.bytes[cy * vp.bytesPerRow + cx * vp.bytesPerPixel];
        _yuvToRgb(yv, u, v, out, o);
        return;
    }
  }

  /// Full range BT.601 (JFIF), what Android cameras deliver.
  static void _yuvToRgb(int y, int u, int v, List<int> out, int o) {
    final d = u - 128, e = v - 128;
    out[o] = _clamp(y + 1.402 * e);
    out[o + 1] = _clamp(y - 0.344136 * d - 0.714136 * e);
    out[o + 2] = _clamp(y + 1.772 * d);
  }

  static int _clamp(double v) => v < 0 ? 0 : (v > 255 ? 255 : v.round());
}

/// Samples the aligned [size] x [size] RGB face crop straight from the raw frame: for every output
/// pixel the inverse similarity transform gives the upright position, which is mapped to the raw
/// buffer and sampled bilinearly. No full-frame colour conversion or rotation is ever performed.
Uint8List warpFaceFromFrame(CameraFrame frame, Affine uprightToCrop, int size) {
  final inv = uprightToCrop.inverse();
  final out = Uint8List(size * size * 3);
  final px = List<int>.filled(12, 0);
  for (var v = 0; v < size; v++) {
    for (var u = 0; u < size; u++) {
      final xu = inv.a * u + inv.b * v + inv.c;
      final yu = inv.d * u + inv.e * v + inv.f;
      final o = (v * size + u) * 3;
      if (xu < -1 || yu < -1 || xu > frame.uprightWidth || yu > frame.uprightHeight) {
        continue; // outside the frame: black, like cv2.warpAffine's border
      }
      final (xr, yr) = frame.toRaw(xu, yu);
      final x0 = xr.floor(), y0 = yr.floor();
      final fx = xr - x0, fy = yr - y0;
      frame._rgbAt(x0, y0, px, 0);
      frame._rgbAt(x0 + 1, y0, px, 3);
      frame._rgbAt(x0, y0 + 1, px, 6);
      frame._rgbAt(x0 + 1, y0 + 1, px, 9);
      for (var c = 0; c < 3; c++) {
        final top = px[c] + (px[3 + c] - px[c]) * fx;
        final bottom = px[6 + c] + (px[9 + c] - px[6 + c]) * fx;
        out[o + c] = (top + (bottom - top) * fy).round();
      }
    }
  }
  return out;
}
