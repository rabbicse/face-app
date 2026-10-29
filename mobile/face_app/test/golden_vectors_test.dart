import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:face_app/core/alignment.dart';
import 'package:face_app/core/embedding.dart';
import 'package:face_app/core/geometry.dart';
import 'package:face_app/core/pose.dart';
import 'package:face_app/core/preprocess.dart';
import 'package:flutter_test/flutter_test.dart';

/// Golden values shared with the Python reference and the web app (shared/test-vectors).
final Map<String, dynamic> vectors =
    jsonDecode(File('../../shared/test-vectors/face-pipeline.json').readAsStringSync()) as Map<String, dynamic>;

List<double> doubles(dynamic l) => [for (final v in l as List) (v as num).toDouble()];

void main() {
  group('alignment', () {
    test('template maps to itself', () {
      final m = alignmentMatrix(arcfaceTemplate112);
      expect(m.values, pairwise(const [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 1e-9));
    });

    for (final (i, c) in (vectors['alignment'] as List).indexed) {
      test('matches the Python reference (case $i)', () {
        final pts = [for (final p in c['landmarks'] as List) Point2((p[0] as num).toDouble(), (p[1] as num).toDouble())];
        final expected = [for (final row in c['matrix'] as List) ...doubles(row)];
        expect(alignmentMatrix(pts).values, pairwise(expected, 1e-4));
      });
    }

    test('inverse round trips', () {
      final m = alignmentMatrix(const [Point2(250, 230), Point2(350, 228), Point2(300, 290), Point2(262, 345), Point2(338, 343)]);
      final p = m.inverse().apply(m.apply(const Point2(10, 20)));
      expect(p.x, closeTo(10, 1e-9));
      expect(p.y, closeTo(20, 1e-9));
    });
  });

  group('pose', () {
    for (final (i, c) in (vectors['pose'] as List).indexed) {
      test('matches the Python reference (case $i)', () {
        Point3 p(String k) {
          final v = doubles(c['points'][k]);
          return Point3(v[0], v[1], v[2]);
        }

        final pose = poseFromLandmarks3d(p('left_eye'), p('right_eye'), p('forehead'), p('chin'));
        expect(pose.yaw, closeTo((c['pose']['yaw'] as num).toDouble(), 1e-3));
        expect(pose.pitch, closeTo((c['pose']['pitch'] as num).toDouble(), 1e-3));
        expect(pose.roll, closeTo((c['pose']['roll'] as num).toDouble(), 1e-3));
        expect(classifyPose(pose)?.wireName, c['label']);
      });
    }

    for (final c in vectors['classify'] as List) {
      test('classifies ${c['pose']} as ${c['label']}', () {
        final pose = HeadPose((c['pose']['yaw'] as num).toDouble(), (c['pose']['pitch'] as num).toDouble(),
            (c['pose']['roll'] as num).toDouble());
        expect(classifyPose(pose)?.wireName, c['label']);
      });
    }
  });

  test('preprocessing matches the reference tensor', () {
    final pre = vectors['preprocess'] as Map<String, dynamic>;
    final rgb = Uint8List.fromList([for (final v in pre['rgb'] as List) v as int]);
    final t = rgbToTensor(rgb, pre['width'] as int, pre['height'] as int);
    expect(t.toList(), pairwise(doubles(pre['tensor']), 1e-6));
  });

  group('embedding math', () {
    final e = vectors['embedding'] as Map<String, dynamic>;

    test('normalisation, cosine and fusion match the reference', () {
      expect(l2Normalize(doubles(e['a'])).toList(), pairwise(doubles(e['a_normalized']), 1e-5));
      expect(cosineSimilarity(doubles(e['a']), doubles(e['b'])), closeTo((e['cosine_ab'] as num).toDouble(), 1e-5));
      expect(fuseEmbeddings([doubles(e['a']), doubles(e['b']), doubles(e['c'])]).toList(),
          pairwise(doubles(e['fused_abc']), 1e-5));
    });

    test('uses the backend wire format', () {
      expect(encodeEmbedding([1.0, -2.0, 3.5]), e['base64_of_1_2_3']);
      final v = Float32List.fromList([for (var i = 0; i < 512; i++) i / 7]);
      expect(decodeEmbedding(encodeEmbedding(v)), v);
    });

    test('rejects zero vectors', () {
      expect(() => l2Normalize([0, 0]), throwsArgumentError);
    });
  });
}

Matcher pairwise(List<double> expected, double tol) => pairwiseCompare<double, double>(
    expected, (e, a) => (a - e).abs() <= tol, 'within $tol of');
