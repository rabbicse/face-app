export const EMBEDDING_DIM = 512;

export function l2Normalize(v: ArrayLike<number>, eps = 1e-10): Float32Array {
    let sum = 0;
    for (let i = 0; i < v.length; i++) {
        sum += v[i] * v[i];
    }
    const norm = Math.sqrt(sum);
    if (!Number.isFinite(norm) || norm < eps) {
        throw new Error("embedding has zero or non-finite norm");
    }
    const out = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) {
        out[i] = v[i] / norm;
    }
    return out;
}

/** Dot product; equals the cosine similarity for L2 normalised vectors. */
export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
    if (a.length !== b.length) {
        throw new Error(`dimension mismatch: ${a.length} vs ${b.length}`);
    }
    let s = 0;
    for (let i = 0; i < a.length; i++) {
        s += a[i] * b[i];
    }
    return s;
}

export function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
    return dot(l2Normalize(a), l2Normalize(b));
}

/** Mean of normalised embeddings, re-normalised (person template / multi frame fusion). */
export function fuseEmbeddings(vectors: readonly ArrayLike<number>[]): Float32Array {
    if (vectors.length === 0) {
        throw new Error("no embeddings to fuse");
    }
    const acc = new Float64Array(vectors[0].length);
    for (const v of vectors) {
        const n = l2Normalize(v);
        for (let i = 0; i < acc.length; i++) {
            acc[i] += n[i];
        }
    }
    return l2Normalize(acc);
}

/** Little endian float32 -> base64 (wire format understood by the backend). */
export function encodeEmbedding(v: ArrayLike<number>): string {
    const f32 = v instanceof Float32Array ? v : Float32Array.from(v);
    const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
    let binary = "";
    for (let i = 0; i < bytes.length; i += 0x8000) {
        binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    }
    return btoa(binary);
}

export function decodeEmbedding(b64: string): Float32Array {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    return new Float32Array(bytes.buffer);
}
