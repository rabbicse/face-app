/**
 * Mapping from MediaPipe Face Landmarker (478-point mesh) to the 5-point layout used by
 * pose estimation. (Alignment for the embedding uses SCRFD keypoints in the worker; the
 * tracker's 5 points are used for pose, ROI and fallback.)
 *
 * Indices (MediaPipe naming is from the SUBJECT's perspective):
 *   468 = right iris centre (image-left in an un-mirrored frame), 473 = left iris centre
 *   33/133 right eye corners, 362/263 left eye corners, 1 = nose tip, 61/291 mouth corners
 */
import { Point, Box, boundingBox } from "./geometry";

export interface NormalizedLandmark { x: number; y: number; z?: number }

export const MP = {
    rightIris: 468, leftIris: 473,
    rightEyeOuter: 33, rightEyeInner: 133,
    leftEyeInner: 362, leftEyeOuter: 263,
    noseTip: 1, mouthRight: 61, mouthLeft: 291,
} as const;

export function mediapipeTo5(lm: readonly NormalizedLandmark[], width: number, height: number): Point[] {
    const px = (i: number): Point => [lm[i].x * width, lm[i].y * height];
    const mid = (a: number, b: number): Point => [(lm[a].x + lm[b].x) / 2 * width, (lm[a].y + lm[b].y) / 2 * height];
    const hasIris = lm.length >= 478;
    return [
        hasIris ? px(MP.rightIris) : mid(MP.rightEyeOuter, MP.rightEyeInner),
        hasIris ? px(MP.leftIris) : mid(MP.leftEyeInner, MP.leftEyeOuter),
        px(MP.noseTip),
        px(MP.mouthRight),
        px(MP.mouthLeft),
    ];
}

export function landmarksBox(lm: readonly NormalizedLandmark[], width: number, height: number): Box {
    return boundingBox(lm.map(p => [p.x * width, p.y * height] as Point));
}

/** True if the face box is well inside the frame (partially visible faces give bad crops). */
export function isInsideFrame(box: Box, width: number, height: number, marginFrac = 0.02): boolean {
    const mx = width * marginFrac, my = height * marginFrac;
    return box.x >= mx && box.y >= my && box.x + box.width <= width - mx && box.y + box.height <= height - my;
}
