import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../config.dart';
import '../core/embedding.dart';
import '../core/registration.dart';
import '../core/pose.dart';

class PersonInfo {
  const PersonInfo({required this.personId, required this.name, this.email});
  final int personId;
  final String name;
  final String? email;

  Map<String, dynamic> toJson() => {'person_id': personId, 'name': name, if (email != null) 'email': email};

  factory PersonInfo.fromJson(Map<String, dynamic> j) =>
      PersonInfo(personId: j['person_id'] as int, name: j['name'] as String, email: j['email'] as String?);
}

class IdentifyResult {
  const IdentifyResult({required this.matched, required this.reason, this.score, this.person});
  final bool matched;
  final String reason;
  final double? score;
  final PersonInfo? person;
}

class ApiException implements Exception {
  ApiException(this.statusCode, this.message);
  final int statusCode;
  final String message;

  @override
  String toString() => 'ApiException($statusCode): $message';
}

/// Talks to /api/v1/embeddings/* - only embeddings are sent, never images.
class FaceApiClient {
  FaceApiClient({String? baseUrl, http.Client? client})
      : baseUrl = baseUrl ?? AppConfig.apiBaseUrl,
        _client = client ?? http.Client();

  final String baseUrl;
  final http.Client _client;

  Future<Map<String, dynamic>> _post(String path, Map<String, dynamic> body) async {
    final res = await _client
        .post(Uri.parse('$baseUrl$path'),
            headers: {'Content-Type': 'application/json', 'Accept': 'application/json'}, body: jsonEncode(body))
        .timeout(const Duration(seconds: 15));
    final decoded = res.body.isEmpty ? <String, dynamic>{} : jsonDecode(res.body) as Map<String, dynamic>;
    if (res.statusCode != 200) {
      throw ApiException(res.statusCode, '${decoded['detail'] ?? res.reasonPhrase}');
    }
    return decoded;
  }

  Future<int> register(PersonInfo person, List<RegistrationSample> samples) async {
    final res = await _post('/api/v1/embeddings/register', {
      'model': AppConfig.embeddingModelId,
      'person': person.toJson(),
      'samples': [
        for (final s in samples)
          {
            'embedding': encodeEmbedding(s.embedding),
            'pose_label': s.poseLabel.wireName,
            'pose': s.pose.toJson(),
            'quality': s.quality,
          }
      ],
    });
    return res['stored_vectors'] as int;
  }

  Future<IdentifyResult> identify(Float32List embedding) async {
    final res = await _post('/api/v1/embeddings/identify', {
      'model': AppConfig.embeddingModelId,
      'embedding': encodeEmbedding(embedding),
      'top_k': 3,
    });
    final person = res['person'];
    return IdentifyResult(
      matched: res['status'] == 0,
      reason: res['reason'] as String,
      score: (res['score'] as num?)?.toDouble(),
      person: person == null ? null : PersonInfo.fromJson(person as Map<String, dynamic>),
    );
  }
}

/// Same as the web app's personIdFromEmail (FNV-1a, 31 bit) - demo only, use real user ids.
int personIdFromEmail(String email) {
  var h = 0x811c9dc5;
  for (final c in email.trim().toLowerCase().runes) {
    h = ((h ^ c) * 0x01000193) & 0xFFFFFFFF;
  }
  final id = h >> 1;
  return id == 0 ? 1 : id;
}
