import 'dart:convert';
import 'dart:typed_data';

import 'package:face_app/api/api_client.dart';
import 'package:face_app/core/embedding.dart';
import 'package:face_app/core/pose.dart';
import 'package:face_app/core/registration.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

void main() {
  test('register sends base64 embeddings with pose metadata', () async {
    late Map<String, dynamic> body;
    final client = FaceApiClient(
      baseUrl: 'http://api',
      client: MockClient((req) async {
        expect(req.url.toString(), 'http://api/api/v1/embeddings/register');
        body = jsonDecode(req.body) as Map<String, dynamic>;
        return http.Response(jsonEncode({'status': 0, 'stored_vectors': 2}), 200);
      }),
    );
    final emb = Float32List.fromList(List.generate(512, (i) => i / 512));
    final stored = await client.register(const PersonInfo(personId: 7, name: 'A', email: 'a@x.io'), [
      RegistrationSample(
          embedding: emb, pose: const HeadPose(1, 2, 3), poseLabel: PoseLabel.turnLeft, quality: 0.8, timestampMs: 0),
    ]);
    expect(stored, 2);
    expect(body['model'], 'w600k_mbf');
    expect(body['person'], {'person_id': 7, 'name': 'A', 'email': 'a@x.io'});
    final sample = (body['samples'] as List).single as Map<String, dynamic>;
    expect(sample['pose_label'], 'turn_left');
    expect(sample['pose'], {'yaw': 1.0, 'pitch': 2.0, 'roll': 3.0});
    expect(decodeEmbedding(sample['embedding'] as String), emb);
  });

  test('identify parses a match and surfaces server errors', () async {
    final ok = FaceApiClient(
      baseUrl: 'http://api',
      client: MockClient((_) async => http.Response(
          jsonEncode({'status': 0, 'reason': 'matched', 'score': 0.81, 'person': {'person_id': 1, 'name': 'B'}}), 200)),
    );
    final res = await ok.identify(Float32List(512)..[0] = 1);
    expect(res.matched, isTrue);
    expect(res.person!.name, 'B');

    final bad = FaceApiClient(
      baseUrl: 'http://api',
      client: MockClient((_) async => http.Response(jsonEncode({'detail': 'bad embedding'}), 422)),
    );
    expect(() => bad.identify(Float32List(512)), throwsA(isA<ApiException>()));
  });

  test('person id matches the web app (FNV-1a)', () {
    // same values as frontend/frs_react/src/face/react/personId.ts and the Python example
    expect(personIdFromEmail('e2e@example.com'), 1135506226);
    expect(personIdFromEmail('Someone@Example.com '), 484615417);
  });
}
