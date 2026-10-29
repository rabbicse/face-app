import 'dart:typed_data';

import 'embedding.dart';
import 'pose.dart';

class PoseRequirement {
  const PoseRequirement(this.label, this.count, this.instruction);
  final PoseLabel label;
  final int count;
  final String instruction;
}

/// Instructions are deliberately side-agnostic: with [RegistrationOptions.flexibleOrder] the
/// left/right (and up/down) samples are accepted in any order and labelled by the measured angle,
/// so the flow works whatever the camera mirroring / angle sign conventions of a device are.
const List<PoseRequirement> fullRegistrationPlan = [
  PoseRequirement(PoseLabel.frontal, 3, 'Look straight at the camera'),
  PoseRequirement(PoseLabel.turnLeft, 2, 'Slowly turn your head to one side, then the other'),
  PoseRequirement(PoseLabel.turnRight, 2, 'Slowly turn your head to one side, then the other'),
  PoseRequirement(PoseLabel.lookUp, 1, 'Tilt your head slightly up, then down'),
  PoseRequirement(PoseLabel.lookDown, 1, 'Tilt your head slightly up, then down'),
];

const List<PoseRequirement> quickRegistrationPlan = [
  PoseRequirement(PoseLabel.frontal, 3, 'Look straight at the camera'),
];

class RegistrationSample {
  const RegistrationSample({
    required this.embedding,
    required this.pose,
    required this.poseLabel,
    required this.quality,
    required this.timestampMs,
  });

  final Float32List embedding;
  final HeadPose pose;
  final PoseLabel poseLabel;
  final double quality;
  final int timestampMs;
}

enum OfferResult { accepted, complete, wrongPose, tooSoon, inconsistent, duplicate }

class RegistrationOptions {
  const RegistrationOptions({
    this.minIntervalMs = 300,
    this.minCrossPoseSimilarity = 0.3,
    this.minSamePoseSimilarity = 0.5,
    this.maxSamePoseSimilarity = 0.995,
    this.flexibleOrder = true,
  });

  final int minIntervalMs;
  final double minCrossPoseSimilarity;
  final double minSamePoseSimilarity;
  final double maxSamePoseSimilarity;

  /// accept the incomplete targets of the current group (left/right, up/down) in any order
  final bool flexibleOrder;
}

const Map<PoseLabel, int> _groups = {
  PoseLabel.frontal: 0,
  PoseLabel.turnLeft: 1,
  PoseLabel.turnRight: 1,
  PoseLabel.lookUp: 2,
  PoseLabel.lookDown: 2,
};

/// Guided multi pose enrolment (port of frontend/frs_react/src/face/core/registration.ts).
/// Frontal samples come first and define the user's baseline pose; the other targets are measured
/// relative to it, which cancels the camera placement (phone held low, tablet on a desk...).
class RegistrationSession {
  RegistrationSession({this.plan = fullRegistrationPlan, this.options = const RegistrationOptions()}) {
    if (plan.isEmpty) throw ArgumentError('registration plan is empty');
  }

  final List<PoseRequirement> plan;
  final RegistrationOptions options;
  final List<RegistrationSample> _samples = [];
  int _lastAccepted = -1 << 40;

  List<RegistrationSample> get collected => List.unmodifiable(_samples);

  int count(PoseLabel label) => _samples.where((s) => s.poseLabel == label).length;

  PoseRequirement? get currentTarget {
    for (final r in plan) {
      if (count(r.label) < r.count) return r;
    }
    return null;
  }

  bool get isComplete => currentTarget == null;

  double get fraction {
    final need = plan.fold<int>(0, (s, r) => s + r.count);
    final have = plan.fold<int>(0, (s, r) => s + (count(r.label) < r.count ? count(r.label) : r.count));
    return have / need;
  }

  /// Incomplete targets that a sample may currently fill.
  List<PoseLabel> get openTargets {
    final target = currentTarget;
    if (target == null) return const [];
    if (!options.flexibleOrder) return [target.label];
    return [
      for (final r in plan)
        if (_groups[r.label] == _groups[target.label] && count(r.label) < r.count) r.label
    ];
  }

  HeadPose relativePose(HeadPose pose) {
    final frontal = _samples.where((s) => s.poseLabel == PoseLabel.frontal).toList();
    if (currentTarget?.label == PoseLabel.frontal || frontal.isEmpty) return pose;
    final yaw0 = frontal.fold<double>(0, (s, f) => s + f.pose.yaw) / frontal.length;
    final pitch0 = frontal.fold<double>(0, (s, f) => s + f.pose.pitch) / frontal.length;
    return HeadPose(pose.yaw - yaw0, pose.pitch - pitch0, pose.roll);
  }

  PoseLabel? _matchingTarget(HeadPose pose) {
    final rel = relativePose(pose);
    for (final label in openTargets) {
      if (matchesPose(rel, label)) return label;
    }
    return null;
  }

  /// Cheap pre-check so callers only compute an embedding when it could be accepted.
  bool wants(HeadPose pose, int timestampMs) =>
      !isComplete && timestampMs - _lastAccepted >= options.minIntervalMs && _matchingTarget(pose) != null;

  OfferResult offer({
    required Float32List embedding,
    required HeadPose pose,
    required double quality,
    required int timestampMs,
  }) {
    if (isComplete) return OfferResult.complete;
    if (timestampMs - _lastAccepted < options.minIntervalMs) return OfferResult.tooSoon;
    final label = _matchingTarget(pose);
    if (label == null) return OfferResult.wrongPose;

    for (final s in _samples.where((s) => s.poseLabel == label)) {
      final sim = dot(s.embedding, embedding);
      if (sim < options.minSamePoseSimilarity) return OfferResult.inconsistent;
      if (sim > options.maxSamePoseSimilarity) return OfferResult.duplicate;
    }
    final frontal = _samples.where((s) => s.poseLabel == PoseLabel.frontal).toList();
    if (label != PoseLabel.frontal && frontal.isNotEmpty) {
      final template = fuseEmbeddings([for (final f in frontal) f.embedding]);
      if (dot(template, embedding) < options.minCrossPoseSimilarity) return OfferResult.inconsistent;
    }

    _samples.add(RegistrationSample(
        embedding: embedding, pose: pose, poseLabel: label, quality: quality, timestampMs: timestampMs));
    _lastAccepted = timestampMs;
    return isComplete ? OfferResult.complete : OfferResult.accepted;
  }

  void reset() {
    _samples.clear();
    _lastAccepted = -1 << 40;
  }
}
