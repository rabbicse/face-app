/**
 * Pixel-level helpers that run identically in the browser worker, in Node (tests / examples)
 * and (ported) in Dart. Everything works on raw RGBA byte buffers so there is no dependency
 * on canvas implementations, which differ slightly between browsers.
 */
import { Affine, invertAffine, ALIGNED_SIZE } from "./geometry";

export interface RgbaImage {
    data: Uint8ClampedArray | Uint8Array;
    width: number;
    height: number;
}

/**
 * Bilinear inverse-mapping warp, equivalent to cv2.warpAffine(src, M, (w, h),
 * flags=INTER_LINEAR, borderMode=BORDER_CONSTANT, borderValue=0).
 * `m` maps SOURCE -> DESTINATION coordinates (same convention as OpenCV).
 */
export function warpAffineRgba(src: RgbaImage, m: Affine, outW: number, outH: number): RgbaImage {
    const inv = invertAffine(m);
    const out = new Uint8ClampedArray(outW * outH * 4);
    const { data, width: W, height: H } = src;
    for (let y = 0; y < outH; y++) {
        // incremental evaluation of the inverse map along the row
        let sx = inv[1] * y + inv[2];
        let sy = inv[4] * y + inv[5];
        for (let x = 0; x < outW; x++, sx += inv[0], sy += inv[3]) {
            const x0 = Math.floor(sx), y0 = Math.floor(sy);
            const fx = sx - x0, fy = sy - y0;
            const o = (y * outW + x) * 4;
            if (x0 >= 0 && y0 >= 0 && x0 + 1 < W && y0 + 1 < H) {
                // fast path: all four neighbours inside
                const i00 = (y0 * W + x0) * 4, i01 = i00 + 4, i10 = i00 + W * 4, i11 = i10 + 4;
                const w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy), w10 = (1 - fx) * fy, w11 = fx * fy;
                out[o] = data[i00] * w00 + data[i01] * w01 + data[i10] * w10 + data[i11] * w11;
                out[o + 1] = data[i00 + 1] * w00 + data[i01 + 1] * w01 + data[i10 + 1] * w10 + data[i11 + 1] * w11;
                out[o + 2] = data[i00 + 2] * w00 + data[i01 + 2] * w01 + data[i10 + 2] * w10 + data[i11 + 2] * w11;
            } else if (x0 >= -1 && y0 >= -1 && x0 < W && y0 < H) {
                // border: missing neighbours count as black (BORDER_CONSTANT 0)
                let r = 0, g = 0, b = 0;
                for (let k = 0; k < 4; k++) {
                    const xx = x0 + (k & 1), yy = y0 + (k >> 1);
                    if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
                    const w = ((k & 1) ? fx : 1 - fx) * ((k >> 1) ? fy : 1 - fy);
                    const i = (yy * W + xx) * 4;
                    r += data[i] * w; g += data[i + 1] * w; b += data[i + 2] * w;
                }
                out[o] = r; out[o + 1] = g; out[o + 2] = b;
            }
            out[o + 3] = 255;
        }
    }
    return { data: out, width: outW, height: outH };
}

/** Aligned 112x112 crop -> ArcFace input: Float32 NCHW, RGB, (x - 127.5) / 127.5. */
export function arcfaceInput(aligned: RgbaImage, out?: Float32Array): Float32Array {
    const n = ALIGNED_SIZE * ALIGNED_SIZE;
    if (aligned.width !== ALIGNED_SIZE || aligned.height !== ALIGNED_SIZE) {
        throw new Error(`arcfaceInput expects ${ALIGNED_SIZE}x${ALIGNED_SIZE}, got ${aligned.width}x${aligned.height}`);
    }
    const t = out ?? new Float32Array(3 * n);
    const d = aligned.data;
    for (let i = 0, p = 0; i < n; i++, p += 4) {
        t[i] = (d[p] - 127.5) / 127.5;
        t[n + i] = (d[p + 1] - 127.5) / 127.5;
        t[2 * n + i] = (d[p + 2] - 127.5) / 127.5;
    }
    return t;
}

/** Horizontally mirrored copy (used for optional flip-test-time augmentation). */
export function flipHorizontal(img: RgbaImage): RgbaImage {
    const { width: w, height: h, data } = img;
    const out = new Uint8ClampedArray(data.length);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const s = (y * w + x) * 4, d = (y * w + (w - 1 - x)) * 4;
            out[d] = data[s]; out[d + 1] = data[s + 1]; out[d + 2] = data[s + 2]; out[d + 3] = data[s + 3];
        }
    }
    return { data: out, width: w, height: h };
}

/** Rec.601 luma as Float32 (0..255). */
export function toGray(img: RgbaImage): Float32Array {
    const n = img.width * img.height;
    const g = new Float32Array(n);
    const d = img.data;
    for (let i = 0, p = 0; i < n; i++, p += 4) g[i] = 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
    return g;
}
