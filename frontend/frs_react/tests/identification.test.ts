import { describe, expect, it } from "vitest";
import { decide, FrameFusion, Gallery } from "@/face/core/identification";
import { FaceQuality } from "@/face/core/quality";
import { base64ToEmbedding, embeddingToBase64, isValidEmbedding, l2Normalize, meanEmbedding, dot } from "@/face/core/vector";
import { loadFixture } from "./helpers";

const Q: FaceQuality = { sharpness: 200, brightness: 130, contrast: 40, interOcularPx: 80, detScore: 0.9 };

function vec(seed: number, noise = 0, salt = 0): Float32Array {
    const v = new Float32Array(512);
    let s = seed, t = 7 + salt;
    for (let i = 0; i < 512; i++) {
        s = (s * 16807) % 2147483647; t = (t * 48271) % 2147483647;
        v[i] = (s / 2147483647 - 0.5) + noise * (t / 2147483647 - 0.5);
    }
    return l2Normalize(v);
}

describe("Gallery", () => {
    it("returns per-person best score, sorted", () => {
        const g = new Gallery<{ name: string }>();
        g.addMany("alice", [vec(1, 0.5, 1), vec(1, 0.5, 2)], { name: "Alice" });
        g.addMany("bob", [vec(2, 0.5, 3)], { name: "Bob" });
        const res = g.search(vec(1, 0.4, 9));
        expect(res[0].personId).toBe("alice");
        expect(res[0].meta?.name).toBe("Alice");
        expect(res).toHaveLength(2);
        expect(res[0].score).toBeGreaterThan(res[1].score);
    });

    it("serialises and restores", () => {
        const g = new Gallery();
        g.addMany("a", [vec(1)], undefined, ["front"]);
        const r = Gallery.fromJSON(JSON.parse(JSON.stringify(g.toJSON())));
        expect(r.size).toBe(1);
        expect(r.search(vec(1))[0].score).toBeCloseTo(1, 5);
        r.remove("a");
        expect(r.size).toBe(0);
    });
});

describe("decide", () => {
    it("accepts above threshold with margin", () => {
        expect(decide([{ personId: "a", score: 0.7 }, { personId: "b", score: 0.3 }]).status).toBe("match");
    });
    it("rejects below threshold", () => {
        const d = decide([{ personId: "a", score: 0.3 }]);
        expect(d.status === "no_match" && d.reason).toBe("below_threshold");
    });
    it("rejects ambiguous results", () => {
        const d = decide([{ personId: "a", score: 0.62 }, { personId: "b", score: 0.6 }]);
        expect(d.status === "no_match" && d.reason).toBe("ambiguous");
    });
    it("handles an empty gallery", () => {
        expect(decide([]).status).toBe("no_match");
    });
});

describe("FrameFusion", () => {
    it("fuses N consistent frames", () => {
        const f = new FrameFusion({ frames: 3, maxAgeMs: 1000, minConsistency: 0.5 });
        f.push(vec(1, 0.5, 1), Q, 0);
        f.push(vec(1, 0.5, 2), Q, 100);
        expect(f.ready()).toBe(false);
        f.push(vec(1, 0.5, 3), Q, 200);
        expect(f.ready()).toBe(true);
        // fused vector is closer to the identity than any single noisy frame
        const fused = dot(f.fused(), vec(1));
        expect(fused).toBeGreaterThan(dot(vec(1, 0.5, 1), vec(1)));
    });
    it("restarts when a different face appears", () => {
        const f = new FrameFusion({ frames: 3, maxAgeMs: 1000, minConsistency: 0.5 });
        f.push(vec(1), Q, 0);
        f.push(vec(1, 0.2, 1), Q, 10);
        f.push(vec(2), Q, 20);
        expect(f.count).toBe(1);
    });
    it("drops stale frames", () => {
        const f = new FrameFusion({ frames: 2, maxAgeMs: 500, minConsistency: 0.5 });
        f.push(vec(1), Q, 0);
        f.push(vec(1, 0.1, 1), Q, 1000);
        expect(f.ready()).toBe(false);
    });
});

describe("vector encoding", () => {
    it("base64 is byte-compatible with Python numpy '<f4'", () => {
        const { base64Probe } = loadFixture();
        expect(Array.from(base64ToEmbedding(base64Probe.b64))).toEqual(Array.from(Float32Array.from(base64Probe.values)));
        expect(embeddingToBase64(Float32Array.from(base64Probe.values))).toBe(base64Probe.b64);
    });
    it("validates embeddings", () => {
        expect(isValidEmbedding(vec(3))).toBe(true);
        expect(isValidEmbedding(new Float32Array(512))).toBe(false);
        expect(isValidEmbedding(vec(3).subarray(0, 128))).toBe(false);
        const bad = vec(3); bad[5] = NaN;
        expect(isValidEmbedding(bad)).toBe(false);
    });
    it("meanEmbedding is normalised", () => {
        const m = meanEmbedding([vec(1), vec(2)]);
        expect(Math.hypot(...m)).toBeCloseTo(1, 5);
    });
});
