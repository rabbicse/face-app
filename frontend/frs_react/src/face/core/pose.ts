import { Point3 } from "./geometry";

/**
 * Head pose in degrees on the non-mirrored camera frame (same convention as Google ML Kit):
 * yaw > 0 face turned towards the image right (the user's own left), pitch > 0 looking up,
 * roll > 0 rotated counter-clockwise in the image.
 */
export interface HeadPose {
    yaw: number;
    pitch: number;
    roll: number;
}

export type PoseLabel = "frontal" | "turn_left" | "turn_right" | "look_up" | "look_down";

interface PoseRange {
    yaw: [number, number];
    pitch: [number, number];
    maxRoll: number;
}

/** Must stay in sync with backend/frs/src/face_pipeline/pose.py and the Flutter app. */
export const POSE_TARGETS: Record<PoseLabel, PoseRange> = {
    frontal: { yaw: [-10, 10], pitch: [-10, 10], maxRoll: 10 },
    turn_left: { yaw: [15, 40], pitch: [-15, 15], maxRoll: 15 },
    turn_right: { yaw: [-40, -15], pitch: [-15, 15], maxRoll: 15 },
    look_up: { yaw: [-12, 12], pitch: [10, 30], maxRoll: 15 },
    look_down: { yaw: [-12, 12], pitch: [-30, -10], maxRoll: 15 },
};

const RAD_TO_DEG = 180 / Math.PI;

/**
 * Pose from 4 points in pixel units (x right, y down, z away from the camera), e.g. the MediaPipe
 * face mesh eyes (image-left / image-right), forehead (10) and chin (152).
 */
export function poseFromLandmarks3d(leftEye: Point3, rightEye: Point3, forehead: Point3, chin: Point3): HeadPose {
    let ax = rightEye.x - leftEye.x, ay = rightEye.y - leftEye.y, az = rightEye.z - leftEye.z;
    const an = Math.hypot(ax, ay, az);
    ax /= an; ay /= an; az /= an;

    let dx = chin.x - forehead.x, dy = chin.y - forehead.y, dz = chin.z - forehead.z;
    const proj = dx * ax + dy * ay + dz * az;
    dx -= ax * proj; dy -= ay * proj; dz -= az * proj;
    const dn = Math.hypot(dx, dy, dz);
    dx /= dn; dy /= dn; dz /= dn;

    // forward = down x xAxis (points towards the camera for a frontal face)
    const fx = dy * az - dz * ay;
    const fy = dz * ax - dx * az;
    const fz = dx * ay - dy * ax;

    return {
        yaw: Math.atan2(fx, -fz) * RAD_TO_DEG,
        pitch: Math.asin(Math.max(-1, Math.min(1, -fy))) * RAD_TO_DEG,
        roll: Math.atan2(-ay, ax) * RAD_TO_DEG,
    };
}

export function matchesPose(pose: HeadPose, label: PoseLabel): boolean {
    const t = POSE_TARGETS[label];
    return pose.yaw >= t.yaw[0] && pose.yaw <= t.yaw[1]
        && pose.pitch >= t.pitch[0] && pose.pitch <= t.pitch[1]
        && Math.abs(pose.roll) <= t.maxRoll;
}

export function classifyPose(pose: HeadPose): PoseLabel | null {
    for (const label of Object.keys(POSE_TARGETS) as PoseLabel[]) {
        if (matchesPose(pose, label)) {
            return label;
        }
    }
    return null;
}
