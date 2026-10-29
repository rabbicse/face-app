import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

const int embeddingDim = 512;

Float32List l2Normalize(List<double> v, {double eps = 1e-10}) {
  var sum = 0.0;
  for (final x in v) {
    sum += x * x;
  }
  final norm = math.sqrt(sum);
  if (!norm.isFinite || norm < eps) {
    throw ArgumentError('embedding has zero or non-finite norm');
  }
  final out = Float32List(v.length);
  for (var i = 0; i < v.length; i++) {
    out[i] = v[i] / norm;
  }
  return out;
}

/// Dot product: the cosine similarity of L2 normalised vectors.
double dot(List<double> a, List<double> b) {
  if (a.length != b.length) throw ArgumentError('dimension mismatch: ${a.length} vs ${b.length}');
  var s = 0.0;
  for (var i = 0; i < a.length; i++) {
    s += a[i] * b[i];
  }
  return s;
}

double cosineSimilarity(List<double> a, List<double> b) => dot(l2Normalize(a), l2Normalize(b));

/// Mean of normalised embeddings, re-normalised (person template / multi frame fusion).
Float32List fuseEmbeddings(List<List<double>> vectors) {
  if (vectors.isEmpty) throw ArgumentError('no embeddings to fuse');
  final acc = Float64List(vectors.first.length);
  for (final v in vectors) {
    final n = l2Normalize(v);
    for (var i = 0; i < acc.length; i++) {
      acc[i] += n[i];
    }
  }
  return l2Normalize(acc);
}

/// Little endian float32 -> base64: the wire format of the backend (same as the web app).
String encodeEmbedding(List<double> v) {
  final f32 = v is Float32List ? v : Float32List.fromList(v);
  final bytes = ByteData(f32.length * 4);
  for (var i = 0; i < f32.length; i++) {
    bytes.setFloat32(i * 4, f32[i], Endian.little);
  }
  return base64Encode(bytes.buffer.asUint8List());
}

Float32List decodeEmbedding(String b64) {
  final bytes = ByteData.sublistView(base64Decode(b64));
  return Float32List.fromList([for (var i = 0; i < bytes.lengthInBytes; i += 4) bytes.getFloat32(i, Endian.little)]);
}
