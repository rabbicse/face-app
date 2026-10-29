import { FACE_CONFIG } from "./config";
import { encodeEmbedding } from "./core/embedding";
import { HeadPose, PoseLabel } from "./core/pose";

export interface PersonInfo {
    person_id: number;
    name: string;
    email?: string;
    phone?: string;
    age?: number;
    city?: string;
    country?: string;
    address?: string;
}

export interface RegisterSampleIn {
    embedding: Float32Array;
    poseLabel: PoseLabel;
    pose: HeadPose;
    quality: number;
}

export interface RegisterResponse {
    status: number;
    message: string;
    stored_vectors: number;
    min_consistency: number;
}

export interface IdentifyResponse {
    status: number;
    reason: "matched" | "below_threshold" | "ambiguous" | "no_enrolled_faces";
    score: number | null;
    person: PersonInfo | null;
    candidates: { person_id: number; score: number }[];
}

export class ApiError extends Error {
    constructor(readonly status: number, message: string) {
        super(message);
    }
}

async function postJson<T>(path: string, body: unknown, signal?: AbortSignal, fetchImpl: typeof fetch = fetch): Promise<T> {
    const res = await fetchImpl(`${FACE_CONFIG.apiBaseUrl}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(body),
        signal,
    });
    if (!res.ok) {
        let detail = res.statusText;
        try {
            detail = (await res.json()).detail ?? detail;
        } catch { /* not json */ }
        throw new ApiError(res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return res.json() as Promise<T>;
}

/** Sends only embeddings (base64 float32, ~2.7 KB each) - never images. */
export function registerEmbeddings(person: PersonInfo, samples: readonly RegisterSampleIn[],
    opts: { signal?: AbortSignal; fetchImpl?: typeof fetch } = {}): Promise<RegisterResponse> {
    return postJson<RegisterResponse>("/api/v1/embeddings/register", {
        model: FACE_CONFIG.embedding.modelId,
        person,
        samples: samples.map(s => ({
            embedding: encodeEmbedding(s.embedding),
            pose_label: s.poseLabel,
            pose: { yaw: s.pose.yaw, pitch: s.pose.pitch, roll: s.pose.roll },
            quality: s.quality,
        })),
    }, opts.signal, opts.fetchImpl);
}

export function identifyEmbedding(embedding: Float32Array,
    opts: { signal?: AbortSignal; fetchImpl?: typeof fetch; topK?: number } = {}): Promise<IdentifyResponse> {
    return postJson<IdentifyResponse>("/api/v1/embeddings/identify", {
        model: FACE_CONFIG.embedding.modelId,
        embedding: encodeEmbedding(embedding),
        top_k: opts.topK ?? 3,
    }, opts.signal, opts.fetchImpl);
}
