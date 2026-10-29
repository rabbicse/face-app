import 'dart:math' as math;
import 'dart:typed_data';

import 'package:face_app/core/embedding.dart';
import 'package:face_app/core/identification.dart';
import 'package:face_app/core/pose.dart';
import 'package:face_app/core/registration.dart';
import 'package:flutter_test/flutter_test.dart';

Float32List identity(int seed) {
  final r = math.Random(seed);
  return l2Normalize([for (var i = 0; i < 512; i++) r.nextDouble() - 0.5]);
}

Float32List noisy(Float32List base, int seed, [double noise = 0.02]) {
  final r = math.Random(seed);
  return l2Normalize([for (final v in base) v + noise * (r.nextDouble() - 0.5)]);
}

const poses = {
  PoseLabel.frontal: HeadPose(0, 0, 0),
  PoseLabel.turnLeft: HeadPose(25, 0, 0),
  PoseLabel.turnRight: HeadPose(-25, 0, 0),
  PoseLabel.lookUp: HeadPose(0, 20, 0),
  PoseLabel.lookDown: HeadPose(0, -20, 0),
};

void main() {
  group('RegistrationSession', () {
    test('collects the full plan', () {
      final s = RegistrationSession();
      final alice = identity(1);
      var t = 0, seed = 100;
      var result = OfferResult.accepted;
      while (!s.isComplete) {
        final label = s.openTargets.first;
        t += 400;
        expect(s.wants(poses[label]!, t), isTrue);
        result = s.offer(embedding: noisy(alice, seed++), pose: poses[label]!, quality: 0.9, timestampMs: t);
      }
      expect(result, OfferResult.complete);
      expect(s.collected, hasLength(9));
      expect(s.fraction, 1.0);
    });

    test('accepts the sides in any order (flexible order)', () {
      final s = RegistrationSession();
      final alice = identity(1);
      for (var i = 0; i < 3; i++) {
        s.offer(embedding: noisy(alice, i), pose: poses[PoseLabel.frontal]!, quality: 1, timestampMs: i * 400);
      }
      expect(s.openTargets, [PoseLabel.turnLeft, PoseLabel.turnRight]);
      // the user turns right first although turn_left comes first in the plan
      expect(s.offer(embedding: noisy(alice, 10), pose: poses[PoseLabel.turnRight]!, quality: 1, timestampMs: 2000),
          OfferResult.accepted);
      expect(s.count(PoseLabel.turnRight), 1);
    });

    test('strict order when flexibleOrder is off', () {
      final s = RegistrationSession(options: const RegistrationOptions(flexibleOrder: false));
      final alice = identity(1);
      for (var i = 0; i < 3; i++) {
        s.offer(embedding: noisy(alice, i), pose: poses[PoseLabel.frontal]!, quality: 1, timestampMs: i * 400);
      }
      expect(s.offer(embedding: noisy(alice, 10), pose: poses[PoseLabel.turnRight]!, quality: 1, timestampMs: 2000),
          OfferResult.wrongPose);
    });

    test('measures poses relative to the frontal baseline', () {
      final s = RegistrationSession(plan: const [
        PoseRequirement(PoseLabel.frontal, 1, ''),
        PoseRequirement(PoseLabel.lookUp, 1, ''),
      ]);
      final alice = identity(1);
      expect(s.offer(embedding: alice, pose: const HeadPose(0, 12, 0), quality: 1, timestampMs: 0), OfferResult.accepted);
      expect(s.wants(const HeadPose(0, 20, 0), 1000), isFalse);
      expect(s.offer(embedding: noisy(alice, 3), pose: const HeadPose(0, 30, 0), quality: 1, timestampMs: 1000),
          OfferResult.complete);
    });

    test('rejects wrong poses, rapid fire, duplicates and other people', () {
      final s = RegistrationSession(plan: quickRegistrationPlan);
      final alice = identity(1);
      expect(s.offer(embedding: alice, pose: poses[PoseLabel.turnLeft]!, quality: 1, timestampMs: 0),
          OfferResult.wrongPose);
      expect(s.offer(embedding: alice, pose: poses[PoseLabel.frontal]!, quality: 1, timestampMs: 0), OfferResult.accepted);
      expect(s.offer(embedding: noisy(alice, 2), pose: poses[PoseLabel.frontal]!, quality: 1, timestampMs: 100),
          OfferResult.tooSoon);
      expect(s.offer(embedding: alice, pose: poses[PoseLabel.frontal]!, quality: 1, timestampMs: 1000),
          OfferResult.duplicate);
      expect(s.offer(embedding: identity(2), pose: poses[PoseLabel.frontal]!, quality: 1, timestampMs: 2000),
          OfferResult.inconsistent);
    });
  });

  group('IdentificationAggregator', () {
    test('fuses a consistent window', () {
      final a = IdentificationAggregator();
      final alice = identity(1);
      expect(a.add(noisy(alice, 1), 0), isNull);
      expect(a.add(noisy(alice, 2), 30), isNull);
      expect(a.add(noisy(alice, 3), 60), isNotNull);
      expect(a.size, 0);
    });

    test('drops stale frames and mixed identities', () {
      final a = IdentificationAggregator();
      final alice = identity(1);
      a.add(noisy(alice, 1), 0);
      a.add(noisy(alice, 2), 10);
      expect(a.add(noisy(alice, 3), 5000), isNull);
      expect(a.size, 1);
      a.reset();
      a.add(noisy(alice, 1), 0);
      a.add(identity(9), 10);
      expect(a.add(noisy(alice, 3), 20), isNull);
    });
  });
}
