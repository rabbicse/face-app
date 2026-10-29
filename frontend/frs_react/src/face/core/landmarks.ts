import { distance, Point, Point3 } from "./geometry";
import { HeadPose, poseFromLandmarks3d, poseFromTransformationMatrix } from "./pose";

export interface NormalizedLandmark {
    x: number;
    y: number;
    z: number;
}

export interface Box {
    x: number;
    y: number;
    width: number;
    height: number;
}

export interface FaceGeometry {
    /** 5 alignment points in source pixels, ArcFace template order */
    points5: Point[];
    box: Box;
    pose: HeadPose;
    /** distance between the eye centres in source pixels */
    interOcular: number;
}

/** MediaPipe face mesh indices (468 / 478 point topology). */
export const MESH = {
    eyeA: [33, 133],
    eyeB: [362, 263],
    irisCenters: [468, 473],
    noseTip: 1,
    mouthA: 61,
    mouthB: 291,
    forehead: 10,
    chin: 152,
} as const;

const toPixels = (l: NormalizedLandmark, w: number, h: number): Point3 => ({ x: l.x * w, y: l.y * h, z: l.z * w });

function eyeCenter(lm: readonly NormalizedLandmark[], corners: readonly number[], w: number, h: number): Point3 {
    const a = toPixels(lm[corners[0]], w, h);
    const b = toPixels(lm[corners[1]], w, h);
    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
    // prefer the iris centre (nearest one) when the model outputs refined 478 point landmarks
    let best: Point3 = mid;
    let bestDist = distance(a, b) * 0.5;
    for (const iris of MESH.irisCenters) {
        if (lm.length > iris) {
            const c = toPixels(lm[iris], w, h);
            const d = distance(c, mid);
            if (d < bestDist) {
                best = c;
                bestDist = d;
            }
        }
    }
    return best;
}

/** Order two points left to right in the image (robust to the mesh index semantics / mirroring). */
function leftRight<T extends Point>(a: T, b: T): [T, T] {
    return a.x <= b.x ? [a, b] : [b, a];
}

/**
 * @param transform optional MediaPipe facial transformation matrix (column-major 4x4); preferred for
 *                  the head pose when available
 */
export function faceGeometryFromMesh(lm: readonly NormalizedLandmark[], width: number, height: number,
    transform?: ArrayLike<number>): FaceGeometry {
    if (lm.length < 468) {
        throw new Error(`expected a 468/478 point face mesh, got ${lm.length} points`);
    }

    const eyeA = eyeCenter(lm, MESH.eyeA, width, height);
    const eyeB = eyeCenter(lm, MESH.eyeB, width, height);
    const [leftEye, rightEye] = leftRight(eyeA, eyeB);
    const [mouthLeft, mouthRight] = leftRight(toPixels(lm[MESH.mouthA], width, height),
        toPixels(lm[MESH.mouthB], width, height));
    const nose = toPixels(lm[MESH.noseTip], width, height);

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let i = 0; i < 468; i++) {
        const x = lm[i].x * width, y = lm[i].y * height;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
    }

    return {
        points5: [leftEye, rightEye, nose, mouthLeft, mouthRight].map(p => ({ x: p.x, y: p.y })),
        box: { x: minX, y: minY, width: maxX - minX, height: maxY - minY },
        pose: transform && transform.length === 16
            ? poseFromTransformationMatrix(transform)
            : poseFromLandmarks3d(leftEye, rightEye, toPixels(lm[MESH.forehead], width, height),
                toPixels(lm[MESH.chin], width, height)),
        interOcular: distance(leftEye, rightEye),
    };
}

