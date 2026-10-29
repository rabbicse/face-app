import { Affine, estimateSimilarityTransform, Point } from "./geometry";

export const ALIGNED_FACE_SIZE = 112;

/**
 * ArcFace / InsightFace 5-point template for a 112x112 crop. Order in image coordinates of a
 * non-mirrored frame: image-left eye, image-right eye, nose tip, image-left and image-right mouth corner.
 */
export const ARCFACE_TEMPLATE_112: readonly Point[] = [
    { x: 38.2946, y: 51.6963 },
    { x: 73.5318, y: 51.5014 },
    { x: 56.0252, y: 71.7366 },
    { x: 41.5493, y: 92.3655 },
    { x: 70.7299, y: 92.2041 },
];

const MOUTH_CENTER_112: Point = {
    x: (ARCFACE_TEMPLATE_112[3].x + ARCFACE_TEMPLATE_112[4].x) / 2,
    y: (ARCFACE_TEMPLATE_112[3].y + ARCFACE_TEMPLATE_112[4].y) / 2,
};

/**
 * Matrix mapping source pixels to the aligned crop.
 * @param landmarks 5 points (template order) or 4 points (eyes, nose, mouth centre) in source pixels
 */
export function alignmentMatrix(landmarks: readonly Point[], size = ALIGNED_FACE_SIZE): Affine {
    let template: readonly Point[];
    if (landmarks.length === 5) {
        template = ARCFACE_TEMPLATE_112;
    } else if (landmarks.length === 4) {
        template = [...ARCFACE_TEMPLATE_112.slice(0, 3), MOUTH_CENTER_112];
    } else {
        throw new Error(`expected 5 or 4 landmarks, got ${landmarks.length}`);
    }
    const scale = size / ALIGNED_FACE_SIZE;
    return estimateSimilarityTransform(landmarks, template.map(p => ({ x: p.x * scale, y: p.y * scale })));
}
