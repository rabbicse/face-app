import 'dart:typed_data';

import 'embedding.dart';

/// Fuses the embeddings of several consecutive good frames into one query (less noise) and rejects
/// windows mixing different faces. Identification has no head pose requirement.
class IdentificationAggregator {
  IdentificationAggregator({this.windowSize = 3, this.maxAgeMs = 1500, this.minAgreement = 0.6});

  final int windowSize;
  final int maxAgeMs;
  final double minAgreement;
  final List<(Float32List, int)> _window = [];

  int get size => _window.length;

  /// Returns the fused embedding once the window is full and consistent, otherwise null.
  Float32List? add(Float32List embedding, int timestampMs) {
    _window.removeWhere((f) => timestampMs - f.$2 > maxAgeMs);
    _window.add((embedding, timestampMs));
    if (_window.length > windowSize) _window.removeAt(0);
    if (_window.length < windowSize) return null;

    final vectors = [for (final f in _window) f.$1];
    for (var i = 0; i < vectors.length; i++) {
      final others = fuseEmbeddings([
        for (var j = 0; j < vectors.length; j++)
          if (j != i) vectors[j]
      ]);
      if (dot(vectors[i], others) < minAgreement) {
        _window.removeAt(0);
        return null;
      }
    }
    _window.clear();
    return fuseEmbeddings(vectors);
  }

  void reset() => _window.clear();
}
