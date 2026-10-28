/**
 * 2-D geometry for face alignment.
 *
 * Pure functions, no DOM. The exact same maths is ported to Dart
 * (mobile/face_app/lib/core/geometry.dart) and Python (examples/python/face_ref.py);
 * tests/geometry.test.ts checks this file against the Python reference.
 */

export type Point = readonly [number, number];

/**
 * 2x3 affine matrix stored row-major: [a, b, c, d, e, f]
 *   x' = a*x + b*y + c
 *   y' = d*x + e*y + f
 * (same layout as OpenCV's warpAffine M and canvas setTransform(a, d, b, e, c, f)).
 */
export type Affine = [number, number, number, number, number, number];

/**
 * ArcFace / InsightFace canonical 5-point template for a 112x112 crop.
 * Order: image-left eye, image-right eye, nose tip, image-left mouth corner, image-right mouth corner.
 * ("image-left" = smaller x in the UN-mirrored camera frame = the subject's right eye.)
 */
export const ARCFACE_TEMPLATE_112: readonly Point[] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

export const ALIGNED_SIZE = 112;

/**
 * Least-squares similarity transform (uniform scale + rotation + translation, no reflection)
 * mapping `src` onto `dst`.
 *
 * In 2-D the Umeyama solution has a closed form: treat points as complex numbers and solve
 * z_dst = w * z_src + t by linear least squares. This is exactly what
 * skimage.transform.SimilarityTransform.estimate returns, without needing an SVD.
 */
export function estimateSimilarity(src: readonly Point[], dst: readonly Point[]): Affine {
    const n = src.length;
    if (n !== dst.length || n < 2) {
        throw new Error(`estimateSimilarity: need >=2 matching points, got ${n}/${dst.length}`);
    }
    let sx = 0, sy = 0, dx = 0, dy = 0;
    for (let i = 0; i < n; i++) {
        sx += src[i][0]; sy += src[i][1];
        dx += dst[i][0]; dy += dst[i][1];
    }
    sx /= n; sy /= n; dx /= n; dy /= n;

    let num_a = 0, num_b = 0, den = 0;
    for (let i = 0; i < n; i++) {
        const px = src[i][0] - sx, py = src[i][1] - sy;
        const qx = dst[i][0] - dx, qy = dst[i][1] - dy;
        num_a += px * qx + py * qy;
        num_b += px * qy - py * qx;
        den += px * px + py * py;
    }
    if (den < 1e-12) throw new Error("estimateSimilarity: degenerate source points");
    const a = num_a / den;  // s*cos(theta)
    const b = num_b / den;  // s*sin(theta)
    return [a, -b, dx - (a * sx - b * sy), b, a, dy - (b * sx + a * sy)];
}

export function applyAffine(m: Affine, p: Point): [number, number] {
    return [m[0] * p[0] + m[1] * p[1] + m[2], m[3] * p[0] + m[4] * p[1] + m[5]];
}

export function invertAffine(m: Affine): Affine {
    const [a, b, c, d, e, f] = m;
    const det = a * e - b * d;
    if (Math.abs(det) < 1e-12) throw new Error("invertAffine: singular matrix");
    const ia = e / det, ib = -b / det, id = -d / det, ie = a / det;
    return [ia, ib, -(ia * c + ib * f), id, ie, -(id * c + ie * f)];
}

/** Composition: (m1 ∘ m2)(p) = m1(m2(p)). */
export function composeAffine(m1: Affine, m2: Affine): Affine {
    return [
        m1[0] * m2[0] + m1[1] * m2[3], m1[0] * m2[1] + m1[1] * m2[4], m1[0] * m2[2] + m1[1] * m2[5] + m1[2],
        m1[3] * m2[0] + m1[4] * m2[3], m1[3] * m2[1] + m1[4] * m2[4], m1[3] * m2[2] + m1[4] * m2[5] + m1[5],
    ];
}

/** Matrix that maps face landmarks to the canonical 112x112 ArcFace crop. */
export function alignmentMatrix(kps5: readonly Point[]): Affine {
    return estimateSimilarity(kps5, ARCFACE_TEMPLATE_112);
}

export function distance(p: Point, q: Point): number {
    return Math.hypot(p[0] - q[0], p[1] - q[1]);
}

/** Inter-ocular distance in pixels for a 5-point set. */
export function interOcular(kps5: readonly Point[]): number {
    return distance(kps5[0], kps5[1]);
}

export interface Box { x: number; y: number; width: number; height: number }

/**
 * Square region of interest around a face box, used to crop a small bitmap that the
 * worker re-detects keypoints on. `scale` > 1 adds context (SCRFD likes some margin).
 */
export function squareRoi(box: Box, scale = 1.6): { x: number; y: number; side: number } {
    const side = Math.max(box.width, box.height) * scale;
    return {
        x: box.x + box.width / 2 - side / 2,
        y: box.y + box.height / 2 - side / 2,
        side,
    };
}

export function boundingBox(points: readonly Point[]): Box {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [x, y] of points) {
        if (x < x0) x0 = x;
        if (y < y0) y0 = y;
        if (x > x1) x1 = x;
        if (y > y1) y1 = y;
    }
    return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
}
