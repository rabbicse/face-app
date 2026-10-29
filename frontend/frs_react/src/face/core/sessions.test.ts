import { describe, expect, it } from "vitest";
import { l2Normalize } from "./embedding";
import { IdentificationAggregator } from "./identification";
import { HeadPose } from "./pose";
import { QUICK_REGISTRATION_PLAN, RegistrationSession } from "./registration";

function rng(seed: number) {
    return () => {
        seed = (seed * 1664525 + 1013904223) % 4294967296;
        return seed / 4294967296 - 0.5;
    };
}

function identity(seed: number): Float32Array {
    const r = rng(seed);
    return l2Normalize(Array.from({ length: 512 }, r));
}

function noisy(base: Float32Array, seed: number, noise = 0.02): Float32Array {
    const r = rng(seed);
    return l2Normalize(Array.from(base, v => v + noise * r()));
}

const POSES: Record<string, HeadPose> = {
    frontal: { yaw: 0, pitch: 0, roll: 0 },
    turn_left: { yaw: 25, pitch: 0, roll: 0 },
    turn_right: { yaw: -25, pitch: 0, roll: 0 },
    look_up: { yaw: 0, pitch: 20, roll: 0 },
    look_down: { yaw: 0, pitch: -20, roll: 0 },
};

describe("RegistrationSession", () => {
    it("walks through the full plan in order", () => {
        const s = new RegistrationSession();
        const alice = identity(1);
        let t = 0, seed = 100, result = "";
        expect(s.currentTarget?.label).toBe("frontal");
        while (!s.isComplete) {
            const label = s.currentTarget!.label;
            t += 400;
            expect(s.wants(POSES[label], t)).toBe(true);
            result = s.offer({ embedding: noisy(alice, seed++), pose: POSES[label], quality: 0.9, timestamp: t });
        }
        expect(result).toBe("complete");
        expect(s.collected).toHaveLength(9);
        expect(s.fraction).toBe(1);
        expect(s.progress.map(p => p.have)).toEqual([3, 2, 2, 1, 1]);
    });

    it("rejects wrong poses, rapid fire and duplicate frames", () => {
        const s = new RegistrationSession();
        const alice = identity(1);
        expect(s.wants(POSES.turn_left, 1000)).toBe(false);
        expect(s.offer({ embedding: alice, pose: POSES.turn_left, quality: 1, timestamp: 1000 })).toBe("wrong_pose");
        expect(s.offer({ embedding: alice, pose: POSES.frontal, quality: 1, timestamp: 1000 })).toBe("accepted");
        expect(s.offer({ embedding: noisy(alice, 2), pose: POSES.frontal, quality: 1, timestamp: 1100 })).toBe("too_soon");
        expect(s.offer({ embedding: alice, pose: POSES.frontal, quality: 1, timestamp: 2000 })).toBe("duplicate");
    });

    it("rejects a different person stepping in", () => {
        const s = new RegistrationSession(QUICK_REGISTRATION_PLAN);
        expect(s.offer({ embedding: identity(1), pose: POSES.frontal, quality: 1, timestamp: 0 })).toBe("accepted");
        expect(s.offer({ embedding: identity(2), pose: POSES.frontal, quality: 1, timestamp: 1000 })).toBe("inconsistent");
        s.reset();
        expect(s.collected).toHaveLength(0);
    });
});

describe("IdentificationAggregator", () => {
    it("fuses a consistent window", () => {
        const a = new IdentificationAggregator();
        const alice = identity(1);
        expect(a.add(noisy(alice, 1), 0)).toBeNull();
        expect(a.add(noisy(alice, 2), 30)).toBeNull();
        const fused = a.add(noisy(alice, 3), 60);
        expect(fused).not.toBeNull();
        expect(a.size).toBe(0);
    });

    it("drops stale frames and inconsistent windows", () => {
        const a = new IdentificationAggregator();
        const alice = identity(1);
        a.add(noisy(alice, 1), 0);
        a.add(noisy(alice, 2), 10);
        expect(a.add(noisy(alice, 3), 5000)).toBeNull(); // the first two expired
        expect(a.size).toBe(1);
        a.reset();
        a.add(noisy(alice, 1), 0);
        a.add(identity(9), 10);
        expect(a.add(noisy(alice, 3), 20)).toBeNull();
    });
});
