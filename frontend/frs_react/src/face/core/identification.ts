/**
 * Identification (1:N). No pose requirements — just quality gating, multi-frame fusion
 * and a decision rule with a margin, which removes most false accepts caused by a single
 * lucky frame.
 */
import { FaceQuality, qualityScore } from "./quality";
import { dot, meanEmbedding } from "./vector";

export interface GalleryEntry<P = unknown> {
    personId: string;
    embedding: Float32Array;
    label?: string;
    meta?: P;
}

export interface Candidate<P = unknown> {
    personId: string;
    score: number;
    meta?: P;
}

/**
 * Tiny in-memory gallery (brute force dot products). 512-d x 10k templates is ~5M MACs
 * (≈2–5 ms in JS), so on-device search is fine for personal / small-office galleries.
 * Bigger galleries go to the server (Qdrant).
 */
export class Gallery<P = unknown> {
    private entries: GalleryEntry<P>[] = [];

    get size(): number { return this.entries.length; }
    get personIds(): string[] { return [...new Set(this.entries.map(e => e.personId))]; }

    add(entry: GalleryEntry<P>): void { this.entries.push(entry); }

    addMany(personId: string, embeddings: Float32Array[], meta?: P, labels?: string[]): void {
        embeddings.forEach((embedding, i) => this.add({ personId, embedding, meta, label: labels?.[i] }));
    }

    remove(personId: string): void { this.entries = this.entries.filter(e => e.personId !== personId); }

    clear(): void { this.entries = []; }

    /** Per-person best template score, sorted descending. */
    search(query: Float32Array, topK = 5): Candidate<P>[] {
        const best = new Map<string, Candidate<P>>();
        for (const e of this.entries) {
            const s = dot(query, e.embedding);
            const cur = best.get(e.personId);
            if (!cur || s > cur.score) best.set(e.personId, { personId: e.personId, score: s, meta: e.meta });
        }
        return [...best.values()].sort((a, b) => b.score - a.score).slice(0, topK);
    }

    toJSON(): { personId: string; label?: string; meta?: P; embedding: number[] }[] {
        return this.entries.map(e => ({ ...e, embedding: Array.from(e.embedding) }));
    }

    static fromJSON<P>(rows: { personId: string; label?: string; meta?: P; embedding: number[] }[]): Gallery<P> {
        const g = new Gallery<P>();
        rows.forEach(r => g.add({ ...r, embedding: Float32Array.from(r.embedding) }));
        return g;
    }
}

export interface DecisionConfig {
    /** cosine similarity needed to accept (w600k_mbf: 0.40–0.50 typical; see docs/THRESHOLDS.md) */
    threshold: number;
    /** best candidate must beat the runner-up by at least this much */
    margin: number;
}

export const DEFAULT_DECISION: DecisionConfig = { threshold: 0.45, margin: 0.06 };

export type Decision<P = unknown> =
    | { status: "match"; candidate: Candidate<P>; runnerUp?: Candidate<P> }
    | { status: "no_match"; best?: Candidate<P>; reason: "below_threshold" | "ambiguous" | "empty_gallery" };

export function decide<P>(candidates: Candidate<P>[], cfg: DecisionConfig = DEFAULT_DECISION): Decision<P> {
    const [best, second] = candidates;
    if (!best) return { status: "no_match", reason: "empty_gallery" };
    if (best.score < cfg.threshold) return { status: "no_match", best, reason: "below_threshold" };
    if (second && best.score - second.score < cfg.margin) return { status: "no_match", best, reason: "ambiguous" };
    return { status: "match", candidate: best, runnerUp: second };
}

export interface FusionConfig {
    /** good frames to fuse into one query */
    frames: number;
    /** frames older than this are dropped (ms) */
    maxAgeMs: number;
    /** frames that disagree with the running mean by more than this are treated as a different face */
    minConsistency: number;
}

export const DEFAULT_FUSION: FusionConfig = { frames: 3, maxAgeMs: 2500, minConsistency: 0.55 };

/**
 * Collects a few good-quality frame embeddings and fuses them (quality-weighted mean).
 * One fused query per attempt instead of one request per video frame: fewer requests,
 * less noise, higher accuracy.
 */
export class FrameFusion {
    private items: { e: Float32Array; w: number; t: number }[] = [];
    constructor(readonly config: FusionConfig = DEFAULT_FUSION) { }

    get count(): number { return this.items.length; }

    push(embedding: Float32Array, quality: FaceQuality, now: number): void {
        this.items = this.items.filter(i => now - i.t <= this.config.maxAgeMs);
        if (this.items.length > 0) {
            const mean = meanEmbedding(this.items.map(i => i.e));
            if (dot(mean, embedding) < this.config.minConsistency) this.items = []; // a different face appeared
        }
        this.items.push({ e: embedding, w: Math.max(0.05, qualityScore(quality)), t: now });
        if (this.items.length > this.config.frames) this.items.shift();
    }

    ready(): boolean { return this.items.length >= this.config.frames; }

    fused(): Float32Array {
        return meanEmbedding(this.items.map(i => i.e), this.items.map(i => i.w));
    }

    reset(): void { this.items = []; }
}
