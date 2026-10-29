import 'dart:math' as math;

/// Head pose in degrees on the upright, non-mirrored camera frame - the Google ML Kit convention:
/// yaw > 0 face turned towards the image right (the user's own left), pitch > 0 looking up,
/// roll > 0 rotated counter-clockwise.
class HeadPose {
  const HeadPose(this.yaw, this.pitch, this.roll);
  final double yaw;
  final double pitch;
  final double roll;

  Map<String, double> toJson() => {'yaw': yaw, 'pitch': pitch, 'roll': roll};

  @override
  String toString() => 'HeadPose(yaw: ${yaw.toStringAsFixed(1)}, pitch: ${pitch.toStringAsFixed(1)}, '
      'roll: ${roll.toStringAsFixed(1)})';
}

enum PoseLabel { frontal, turnLeft, turnRight, lookUp, lookDown }

extension PoseLabelWire on PoseLabel {
  /// Name used by the backend and the web app.
  String get wireName => const {
        PoseLabel.frontal: 'frontal',
        PoseLabel.turnLeft: 'turn_left',
        PoseLabel.turnRight: 'turn_right',
        PoseLabel.lookUp: 'look_up',
        PoseLabel.lookDown: 'look_down',
      }[this]!;

  static PoseLabel? fromWire(String? name) {
    for (final l in PoseLabel.values) {
      if (l.wireName == name) return l;
    }
    return null;
  }
}

class PoseRange {
  const PoseRange(this.yawMin, this.yawMax, this.pitchMin, this.pitchMax, this.maxRoll);
  final double yawMin, yawMax, pitchMin, pitchMax, maxRoll;
}

/// Must stay in sync with backend/frs/src/face_pipeline/pose.py and frontend/.../core/pose.ts.
const Map<PoseLabel, PoseRange> poseTargets = {
  PoseLabel.frontal: PoseRange(-12, 12, -15, 15, 10),
  PoseLabel.turnLeft: PoseRange(15, 40, -15, 15, 15),
  PoseLabel.turnRight: PoseRange(-40, -15, -15, 15, 15),
  PoseLabel.lookUp: PoseRange(-12, 12, 10, 30, 15),
  PoseLabel.lookDown: PoseRange(-12, 12, -30, -10, 15),
};

bool matchesPose(HeadPose pose, PoseLabel label) {
  final t = poseTargets[label]!;
  return pose.yaw >= t.yawMin &&
      pose.yaw <= t.yawMax &&
      pose.pitch >= t.pitchMin &&
      pose.pitch <= t.pitchMax &&
      pose.roll.abs() <= t.maxRoll;
}

PoseLabel? classifyPose(HeadPose pose) {
  for (final label in PoseLabel.values) {
    if (matchesPose(pose, label)) return label;
  }
  return null;
}

class Point3 {
  const Point3(this.x, this.y, this.z);
  final double x, y, z;
}

/// Pose from 4 points in pixels (x right, y down, z away from the camera). Reference
/// implementation shared with Python/TypeScript; the app itself uses ML Kit's Euler angles.
HeadPose poseFromLandmarks3d(Point3 leftEye, Point3 rightEye, Point3 forehead, Point3 chin) {
  var ax = rightEye.x - leftEye.x, ay = rightEye.y - leftEye.y, az = rightEye.z - leftEye.z;
  final an = math.sqrt(ax * ax + ay * ay + az * az);
  ax /= an;
  ay /= an;
  az /= an;
  var dx = chin.x - forehead.x, dy = chin.y - forehead.y, dz = chin.z - forehead.z;
  final proj = dx * ax + dy * ay + dz * az;
  dx -= ax * proj;
  dy -= ay * proj;
  dz -= az * proj;
  final dn = math.sqrt(dx * dx + dy * dy + dz * dz);
  dx /= dn;
  dy /= dn;
  dz /= dn;
  final fx = dy * az - dz * ay;
  final fy = dz * ax - dx * az;
  final fz = dx * ay - dy * ax;
  const deg = 180 / math.pi;
  return HeadPose(
    math.atan2(fx, -fz) * deg,
    math.asin((-fy).clamp(-1.0, 1.0)) * deg,
    math.atan2(-ay, ax) * deg,
  );
}
