import 'dart:typed_data';

import 'package:face_app/core/geometry.dart';
import 'package:face_app/vision/camera_frame.dart';
import 'package:flutter_test/flutter_test.dart';

/// Upright test image: R encodes x, G encodes y, B constant.
const w = 64, h = 48;
List<int> uprightRgb(int x, int y) => [x * 3, y * 4, 90];

/// Builds a raw frame whose upright view (after [rotation]) is the test image.
CameraFrame makeFrame(PixelLayout layout, int rotation) {
  final rawW = rotation % 180 == 0 ? w : h, rawH = rotation % 180 == 0 ? h : w;
  final probe = CameraFrame(width: rawW, height: rawH, layout: layout, planes: const [], rotation: rotation);
  // raw pixel -> upright pixel: invert toRaw by searching the 4 rotations analytically
  List<int> rawRgb(int xr, int yr) {
    for (var yu = 0; yu < h; yu++) {
      for (var xu = 0; xu < w; xu++) {
        final (x, y) = probe.toRaw(xu.toDouble(), yu.toDouble());
        if (x == xr && y == yr) return uprightRgb(xu, yu);
      }
    }
    throw StateError('unmapped pixel');
  }

  final rgb = List.generate(rawH, (y) => List.generate(rawW, (x) => rawRgb(x, y)));
  switch (layout) {
    case PixelLayout.bgra8888:
      final b = Uint8List(rawW * rawH * 4);
      for (var y = 0; y < rawH; y++) {
        for (var x = 0; x < rawW; x++) {
          final i = (y * rawW + x) * 4, p = rgb[y][x];
          b[i] = p[2];
          b[i + 1] = p[1];
          b[i + 2] = p[0];
          b[i + 3] = 255;
        }
      }
      return CameraFrame(width: rawW, height: rawH, layout: layout, rotation: rotation, planes: [FramePlane(b, rawW * 4, 4)]);
    case PixelLayout.nv21:
    case PixelLayout.yuv420:
      final yPlane = Uint8List(rawW * rawH);
      final u = Uint8List((rawW ~/ 2) * (rawH ~/ 2)), v = Uint8List((rawW ~/ 2) * (rawH ~/ 2));
      for (var y = 0; y < rawH; y++) {
        for (var x = 0; x < rawW; x++) {
          final p = rgb[y][x];
          yPlane[y * rawW + x] = (0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).round();
          if (x.isEven && y.isEven) {
            final ci = (y ~/ 2) * (rawW ~/ 2) + x ~/ 2;
            u[ci] = (128 - 0.168736 * p[0] - 0.331264 * p[1] + 0.5 * p[2]).round().clamp(0, 255);
            v[ci] = (128 + 0.5 * p[0] - 0.418688 * p[1] - 0.081312 * p[2]).round().clamp(0, 255);
          }
        }
      }
      if (layout == PixelLayout.yuv420) {
        return CameraFrame(width: rawW, height: rawH, layout: layout, rotation: rotation, planes: [
          FramePlane(yPlane, rawW),
          FramePlane(u, rawW ~/ 2),
          FramePlane(v, rawW ~/ 2),
        ]);
      }
      final nv21 = Uint8List(rawW * rawH * 3 ~/ 2)..setAll(0, yPlane);
      for (var i = 0; i < u.length; i++) {
        nv21[rawW * rawH + 2 * i] = v[i];
        nv21[rawW * rawH + 2 * i + 1] = u[i];
      }
      return CameraFrame(width: rawW, height: rawH, layout: layout, rotation: rotation, planes: [FramePlane(nv21, rawW)]);
  }
}

void main() {
  for (final layout in PixelLayout.values) {
    for (final rotation in [0, 90, 180, 270]) {
      test('warps $layout frames with rotation $rotation in upright coordinates', () {
        final frame = makeFrame(layout, rotation);
        expect(frame.uprightWidth, w);
        expect(frame.uprightHeight, h);
        // crop 16x16 starting at upright (20, 10), scale 1
        const m = Affine(1, 0, -20, 0, 1, -10);
        final crop = warpFaceFromFrame(frame, m, 16);
        // chroma is subsampled for YUV, so allow some error there
        final tol = layout == PixelLayout.bgra8888 ? 1 : 12;
        for (final (u, v) in [(0, 0), (15, 0), (0, 15), (7, 9), (15, 15)]) {
          final want = uprightRgb(20 + u, 10 + v);
          final o = (v * 16 + u) * 3;
          for (var c = 0; c < 3; c++) {
            expect((crop[o + c] - want[c]).abs(), lessThanOrEqualTo(tol),
                reason: 'pixel ($u,$v) channel $c: got ${crop[o + c]} want ${want[c]}');
          }
        }
      });
    }
  }

  test('rotation means "rotate the raw buffer clockwise to make it upright" (ML Kit semantics)', () {
    // raw 4x2 landscape buffer; rotating it 90 degrees clockwise moves the raw top-left corner to the
    // upright top-right corner, rotating it 270 degrees moves it to the upright bottom-left corner
    const r90 = CameraFrame(width: 4, height: 2, layout: PixelLayout.bgra8888, planes: [], rotation: 90);
    expect(r90.uprightWidth, 2);
    expect(r90.toRaw(1, 0), (0.0, 0.0));
    expect(r90.toRaw(0, 3), (3.0, 1.0)); // raw bottom-right -> upright bottom-left
    const r270 = CameraFrame(width: 4, height: 2, layout: PixelLayout.bgra8888, planes: [], rotation: 270);
    expect(r270.toRaw(0, 3), (0.0, 0.0));
    expect(r270.toRaw(1, 0), (3.0, 1.0));
    const r180 = CameraFrame(width: 4, height: 2, layout: PixelLayout.bgra8888, planes: [], rotation: 180);
    expect(r180.toRaw(3, 1), (0.0, 0.0));
  });

  test('pixels outside the frame stay black', () {
    final frame = makeFrame(PixelLayout.bgra8888, 0);
    final crop = warpFaceFromFrame(frame, const Affine(1, 0, 100, 0, 1, 100), 4);
    expect(crop.every((b) => b == 0), isTrue);
  });

  test('bilinear sampling interpolates between pixels', () {
    final frame = makeFrame(PixelLayout.bgra8888, 0);
    final crop = warpFaceFromFrame(frame, const Affine(1, 0, -10.5, 0, 1, -10), 1);
    expect(crop[0], (10.5 * 3).round()); // R encodes x
  });
}
