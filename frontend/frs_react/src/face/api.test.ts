import { describe, expect, it, vi } from "vitest";
import { ApiError, identifyEmbedding, registerEmbeddings } from "./api";
import { decodeEmbedding } from "./core/embedding";

function fakeFetch(status: number, body: unknown) {
    return vi.fn<typeof fetch>(async () =>
        new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
}

describe("api client", () => {
    it("sends base64 embeddings with pose metadata on registration", async () => {
        const fetchImpl = fakeFetch(200, { status: 0, message: "ok", stored_vectors: 2, min_consistency: 0.9 });
        const embedding = Float32Array.from({ length: 512 }, (_, i) => i / 512);
        await registerEmbeddings({ person_id: 7, name: "A" },
            [{ embedding, poseLabel: "frontal", pose: { yaw: 1, pitch: 2, roll: 3 }, quality: 0.8 }],
            { fetchImpl: fetchImpl });

        const [url, init] = fetchImpl.mock.calls[0];
        expect(String(url)).toMatch(/\/api\/v1\/embeddings\/register$/);
        const body = JSON.parse(String(init?.body));
        expect(body.model).toBe("w600k_mbf");
        expect(body.samples[0].pose_label).toBe("frontal");
        expect(Array.from(decodeEmbedding(body.samples[0].embedding))).toEqual(Array.from(embedding));
    });

    it("returns the identification result", async () => {
        const fetchImpl = fakeFetch(200, { status: 0, reason: "matched", score: 0.8, person: { person_id: 1, name: "B" }, candidates: [] });
        const res = await identifyEmbedding(new Float32Array(512).fill(1), { fetchImpl: fetchImpl });
        expect(res.person?.name).toBe("B");
    });

    it("raises ApiError with the server detail", async () => {
        const fetchImpl = fakeFetch(422, { detail: "embedding must have 512 dimensions" });
        await expect(identifyEmbedding(new Float32Array(3), { fetchImpl: fetchImpl }))
            .rejects.toEqual(new ApiError(422, "embedding must have 512 dimensions"));
    });
});
