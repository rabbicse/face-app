/**
 * RGBA pixels (ImageData layout) -> float32 NCHW tensor in [-1, 1] ((x - 127.5) / 127.5, RGB),
 * the input expected by InsightFace ArcFace models.
 * @param flip also mirror horizontally (test time augmentation)
 */
export function rgbaToTensor(rgba: ArrayLike<number>, width: number, height: number,
    out = new Float32Array(3 * width * height), offset = 0, flip = false): Float32Array {
    const plane = width * height;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const src = (y * width + (flip ? width - 1 - x : x)) * 4;
            const dst = offset + y * width + x;
            out[dst] = (rgba[src] - 127.5) / 127.5;
            out[dst + plane] = (rgba[src + 1] - 127.5) / 127.5;
            out[dst + 2 * plane] = (rgba[src + 2] - 127.5) / 127.5;
        }
    }
    return out;
}

/** BT.601 luma of RGBA pixels. */
export function toGray(rgba: ArrayLike<number>, width: number, height: number): Float32Array {
    const gray = new Float32Array(width * height);
    for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
        gray[i] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
    }
    return gray;
}

/** Variance of the 3x3 Laplacian: a cheap focus / motion blur measure (higher = sharper). */
export function laplacianVariance(gray: ArrayLike<number>, width: number, height: number): number {
    let sum = 0, sumSq = 0, n = 0;
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const i = y * width + x;
            const v = gray[i - width] + gray[i + width] + gray[i - 1] + gray[i + 1] - 4 * gray[i];
            sum += v;
            sumSq += v * v;
            n++;
        }
    }
    if (n === 0) {
        return 0;
    }
    const mean = sum / n;
    return sumSq / n - mean * mean;
}

export function meanValue(values: ArrayLike<number>): number {
    let s = 0;
    for (let i = 0; i < values.length; i++) {
        s += values[i];
    }
    return values.length ? s / values.length : 0;
}
