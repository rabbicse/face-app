/**
 * Head pose (yaw / pitch / roll, degrees) from the 5 face keypoints.
 *
 * One estimator is shared by every platform (web: MediaPipe points, mobile: ML Kit points,
 * offline: SCRFD points) so enrollment thresholds mean the same thing everywhere.
 *
 * Method: weak-perspective fit of a generic 3-D 5-point face model.
 *   x_2d ≈ s · R[0:2] · X_3d + t   →  linear least squares for the 2x3 block, then
 *   orthonormalise the two rows, r3 = r1 × r2, and read Euler angles from R = Rz·Rx·Ry.
 *
 * Conventions (UN-mirrored camera image, x right, y down):
 *   yaw   > 0  → subject turned to THEIR left  (nose moves toward image right)
 *   pitch > 0  → subject looking UP           (nose moves toward image top)
 *   roll  > 0  → head tilted clockwise in the image (image-right eye lower)
 */
import { Point } from "./geometry";

export interface HeadPose { yaw: number; pitch: number; roll: number }

/**
 * Generic 3-D model (mm), right-handed: x right, y down, z AWAY from the camera.
 * Order matches the 5-point keypoint order (image-left eye, image-right eye, nose,
 * image-left mouth, image-right mouth). Frontal projection reproduces the ArcFace template
 * proportions; depth values follow common anthropometric face models.
 */
export const FACE_MODEL_3D: readonly (readonly [number, number, number])[] = [
    [-31.5, 0.0, 0.0],
    [31.5, 0.0, 0.0],
    [0.0, 35.8, -25.0],
    [-26.1, 72.6, -5.0],
    [26.1, 72.6, -5.0],
];

const RAD2DEG = 180 / Math.PI;

function inv3(m: number[][]): number[][] {
    const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2];
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const det = a * A + b * B + c * C;
    if (Math.abs(det) < 1e-12) throw new Error("singular 3x3");
    return [
        [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
        [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
        [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
    ];
}

// Pre-compute pseudo-inverse pieces of the centred model (constant).
const MODEL_MEAN = [0, 1, 2].map(k => FACE_MODEL_3D.reduce((s, p) => s + p[k], 0) / FACE_MODEL_3D.length);
const MODEL_C = FACE_MODEL_3D.map(p => [p[0] - MODEL_MEAN[0], p[1] - MODEL_MEAN[1], p[2] - MODEL_MEAN[2]]);
const XtX_INV = inv3([0, 1, 2].map(r => [0, 1, 2].map(c => MODEL_C.reduce((s, p) => s + p[r] * p[c], 0))));

export function estimateHeadPose(kps5: readonly Point[]): HeadPose {
    if (kps5.length !== 5) throw new Error("estimateHeadPose needs 5 points");
    const mx = kps5.reduce((s, p) => s + p[0], 0) / 5;
    const my = kps5.reduce((s, p) => s + p[1], 0) / 5;

    // Xc^T * x (3x2)
    const xtY = [[0, 0], [0, 0], [0, 0]];
    for (let i = 0; i < 5; i++) {
        const u = kps5[i][0] - mx, v = kps5[i][1] - my;
        for (let k = 0; k < 3; k++) {
            xtY[k][0] += MODEL_C[i][k] * u;
            xtY[k][1] += MODEL_C[i][k] * v;
        }
    }
    // A^T = (Xc^T Xc)^-1 Xc^T x  → rows of A (2x3)
    const r1 = [0, 1, 2].map(k => XtX_INV[k][0] * xtY[0][0] + XtX_INV[k][1] * xtY[1][0] + XtX_INV[k][2] * xtY[2][0]);
    const r2 = [0, 1, 2].map(k => XtX_INV[k][0] * xtY[0][1] + XtX_INV[k][1] * xtY[1][1] + XtX_INV[k][2] * xtY[2][1]);

    // orthonormalise (symmetric Gram-Schmidt keeps both rows equally weighted)
    const n1 = Math.hypot(...r1), n2 = Math.hypot(...r2);
    let a = r1.map(v => v / n1), b = r2.map(v => v / n2);
    const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const a2 = a.map((v, k) => v - 0.5 * dot * b[k]);
    const b2 = b.map((v, k) => v - 0.5 * dot * a[k]);
    a = a2.map(v => v / Math.hypot(...a2));
    b = b2.map(v => v / Math.hypot(...b2));
    const c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

    // R = Rz(roll) · Rx(phi) · Ry(theta); rows: a, b, c
    const phi = Math.asin(Math.max(-1, Math.min(1, c[1])));
    const theta = Math.atan2(-c[0], c[2]);
    const psi = Math.atan2(-a[1], b[1]);
    return { yaw: -theta * RAD2DEG, pitch: -phi * RAD2DEG, roll: psi * RAD2DEG };
}

/** Builds the rotation used by the estimator (for tests / synthetic data). */
export function rotationFromPose({ yaw, pitch, roll }: HeadPose): number[][] {
    const t = -yaw / RAD2DEG, p = -pitch / RAD2DEG, r = roll / RAD2DEG;
    const Ry = [[Math.cos(t), 0, Math.sin(t)], [0, 1, 0], [-Math.sin(t), 0, Math.cos(t)]];
    const Rx = [[1, 0, 0], [0, Math.cos(p), -Math.sin(p)], [0, Math.sin(p), Math.cos(p)]];
    const Rz = [[Math.cos(r), -Math.sin(r), 0], [Math.sin(r), Math.cos(r), 0], [0, 0, 1]];
    const mul = (A: number[][], B: number[][]) => A.map(row => [0, 1, 2].map(j => row.reduce((s, v, k) => s + v * B[k][j], 0)));
    return mul(Rz, mul(Rx, Ry));
}

/** Perspective projection of the model at a given pose (for tests / demos). */
export function projectModel(pose: HeadPose, opts: { focal?: number; distance?: number; cx?: number; cy?: number } = {}): Point[] {
    const { focal = 600, distance = 450, cx = 320, cy = 240 } = opts;
    const R = rotationFromPose(pose);
    return FACE_MODEL_3D.map(([x, y, z]) => {
        const X = x - MODEL_MEAN[0], Y = y - MODEL_MEAN[1], Z = z - MODEL_MEAN[2];
        const px = R[0][0] * X + R[0][1] * Y + R[0][2] * Z;
        const py = R[1][0] * X + R[1][1] * Y + R[1][2] * Z;
        const pz = R[2][0] * X + R[2][1] * Y + R[2][2] * Z + distance;
        return [cx + focal * px / pz, cy + focal * py / pz] as Point;
    });
}
