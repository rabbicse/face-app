import 'dart:typed_data';

/// Interleaved RGB bytes -> float32 NCHW tensor in [-1, 1] ((x - 127.5) / 127.5), the input of
/// InsightFace ArcFace models. Writes into [out] at [offset]; [flip] mirrors horizontally (TTA).
Float32List rgbToTensor(Uint8List rgb, int width, int height,
    {Float32List? out, int offset = 0, bool flip = false}) {
  final plane = width * height;
  final dst = out ?? Float32List(3 * plane);
  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      final s = (y * width + (flip ? width - 1 - x : x)) * 3;
      final d = offset + y * width + x;
      dst[d] = (rgb[s] - 127.5) / 127.5;
      dst[d + plane] = (rgb[s + 1] - 127.5) / 127.5;
      dst[d + 2 * plane] = (rgb[s + 2] - 127.5) / 127.5;
    }
  }
  return dst;
}

/// BT.601 luma of interleaved RGB.
Float32List rgbToGray(Uint8List rgb, int width, int height) {
  final gray = Float32List(width * height);
  for (var i = 0, j = 0; i < gray.length; i++, j += 3) {
    gray[i] = 0.299 * rgb[j] + 0.587 * rgb[j + 1] + 0.114 * rgb[j + 2];
  }
  return gray;
}

/// Variance of the 3x3 Laplacian (focus / motion blur measure, higher is sharper).
double laplacianVariance(List<double> gray, int width, int height) {
  var sum = 0.0, sumSq = 0.0;
  var n = 0;
  for (var y = 1; y < height - 1; y++) {
    for (var x = 1; x < width - 1; x++) {
      final i = y * width + x;
      final v = gray[i - width] + gray[i + width] + gray[i - 1] + gray[i + 1] - 4 * gray[i];
      sum += v;
      sumSq += v * v;
      n++;
    }
  }
  if (n == 0) return 0;
  final mean = sum / n;
  return sumSq / n - mean * mean;
}

double meanValue(List<double> values) {
  if (values.isEmpty) return 0;
  var s = 0.0;
  for (final v in values) {
    s += v;
  }
  return s / values.length;
}
