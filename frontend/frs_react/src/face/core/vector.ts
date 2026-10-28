/** Embedding vector utilities. Embeddings are always L2-normalised, so cosine == dot. */

export const EMBEDDING_DIM = 512;

/**
 * Identifies the embedding space. Vectors from different models are NOT comparable;
 * the server rejects mismatches. Bump the suffix if preprocessing ever changes.
 */
export const EMBEDDING_MODEL_ID = "w600k_mbf+scrfd5@1";

export function l2Normalize(v: Float32Array): Float32Array {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    const n = Math.sqrt(s);
    if (n < 1e-12) throw new Error("cannot normalise a zero vector");
    const out = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
    return out;
}

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
    if (a.length !== b.length) throw new Error(`dimension mismatch ${a.length} vs ${b.length}`);
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
    let na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { na += a[i] * a[i]; nb += b[i] * b[i]; }
    return dot(a, b) / (Math.sqrt(na * nb) || 1);
}

/** Normalised mean of several normalised embeddings (template / multi-frame fusion). */
export function meanEmbedding(vs: readonly Float32Array[], weights?: readonly number[]): Float32Array {
    if (vs.length === 0) throw new Error("meanEmbedding of empty list");
    const out = new Float32Array(vs[0].length);
    vs.forEach((v, k) => {
        const w = weights ? weights[k] : 1;
        for (let i = 0; i < v.length; i++) out[i] += w * v[i];
    });
    return l2Normalize(out);
}

export function isValidEmbedding(v: ArrayLike<number>, dim = EMBEDDING_DIM): boolean {
    if (v.length !== dim) return false;
    let s = 0;
    for (let i = 0; i < v.length; i++) {
        if (!Number.isFinite(v[i])) return false;
        s += v[i] * v[i];
    }
    return Math.abs(Math.sqrt(s) - 1) < 1e-3;
}

/** Little-endian float32 -> base64 (2048 bytes -> 2732 chars for 512-d). */
export function embeddingToBase64(v: Float32Array): string {
    const bytes = new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
    if (typeof Buffer !== "undefined") return Buffer.from(bytes).toString("base64");
    let bin = "";
    for (let i = 0; i < bytes.length; i += 0x8000) {
        bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    }
    return btoa(bin);
}

export function base64ToEmbedding(b64: string): Float32Array {
    let bytes: Uint8Array;
    if (typeof Buffer !== "undefined") {
        bytes = new Uint8Array(Buffer.from(b64, "base64"));
    } else {
        const bin = atob(b64);
        bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    }
    if (bytes.byteLength % 4 !== 0) throw new Error("invalid embedding payload");
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
}
