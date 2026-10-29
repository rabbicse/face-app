export interface Point {
    x: number;
    y: number;
}

export interface Point3 extends Point {
    z: number;
}

/** Row major 2x3 affine matrix [a, b, c, d, e, f]: x' = a*x + b*y + c, y' = d*x + e*y + f */
export type Affine = [number, number, number, number, number, number];

/**
 * Least squares non-reflective similarity transform mapping `src` onto `dst`
 * (closed form of Umeyama for 2D, identical to skimage's SimilarityTransform).
 */
export function estimateSimilarityTransform(src: readonly Point[], dst: readonly Point[]): Affine {
    if (src.length !== dst.length || src.length < 2) {
        throw new Error(`expected two point sets of equal length >= 2, got ${src.length} and ${dst.length}`);
    }
    const n = src.length;
    let sx = 0, sy = 0, dx = 0, dy = 0;
    for (let i = 0; i < n; i++) {
        sx += src[i].x; sy += src[i].y;
        dx += dst[i].x; dy += dst[i].y;
    }
    sx /= n; sy /= n; dx /= n; dy /= n;

    let denom = 0, aNum = 0, bNum = 0;
    for (let i = 0; i < n; i++) {
        const px = src[i].x - sx, py = src[i].y - sy;
        const qx = dst[i].x - dx, qy = dst[i].y - dy;
        denom += px * px + py * py;
        aNum += px * qx + py * qy;
        bNum += px * qy - py * qx;
    }
    if (denom < 1e-12) {
        throw new Error("source points are degenerate");
    }
    const a = aNum / denom;
    const b = bNum / denom;
    return [a, -b, dx - (a * sx - b * sy), b, a, dy - (b * sx + a * sy)];
}

export function applyAffine(m: Affine, p: Point): Point {
    return { x: m[0] * p.x + m[1] * p.y + m[2], y: m[3] * p.x + m[4] * p.y + m[5] };
}

export function distance(a: Point, b: Point): number {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

export function midpoint(a: Point, b: Point): Point {
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}
