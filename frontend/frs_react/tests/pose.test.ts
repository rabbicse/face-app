import { describe, expect, it } from "vitest";
import { estimateHeadPose, projectModel } from "@/face/core/pose";
import { ARCFACE_TEMPLATE_112, Point } from "@/face/core/geometry";

function rng(seed: number) {
    return () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
}

describe("estimateHeadPose", () => {
    it("is ~0 for the frontal ArcFace template", () => {
        const p = estimateHeadPose(ARCFACE_TEMPLATE_112);
        expect(Math.abs(p.yaw)).toBeLessThan(2);
        expect(Math.abs(p.pitch)).toBeLessThan(2);
        expect(Math.abs(p.roll)).toBeLessThan(1);
    });

    it("recovers synthetic poses under perspective projection within a few degrees", () => {
        for (const yaw of [-30, -20, -10, 0, 10, 20, 30]) {
            for (const pitch of [-20, -10, 0, 10, 20]) {
                for (const roll of [-10, 0, 10]) {
                    const est = estimateHeadPose(projectModel({ yaw, pitch, roll }));
                    expect(Math.abs(est.yaw - yaw)).toBeLessThan(3.5);
                    expect(Math.abs(est.pitch - pitch)).toBeLessThan(3.5);
                    expect(Math.abs(est.roll - roll)).toBeLessThan(2.5);
                }
            }
        }
    });

    it("stays usable with realistic landmark noise (σ = 1.5 px at ~110 px IOD)", () => {
        const r = rng(7);
        let worst = 0;
        for (let i = 0; i < 200; i++) {
            const pose = { yaw: (r() - 0.5) * 50, pitch: (r() - 0.5) * 36, roll: (r() - 0.5) * 20 };
            const pts = projectModel(pose, { distance: 350 }).map(([x, y]) => [x + (r() - 0.5) * 3, y + (r() - 0.5) * 3] as Point);
            const est = estimateHeadPose(pts);
            worst = Math.max(worst, Math.abs(est.yaw - pose.yaw), Math.abs(est.pitch - pose.pitch));
        }
        expect(worst).toBeLessThan(8);
    });

    it("follows the documented sign conventions", () => {
        // subject turns to THEIR left -> nose moves to image right -> yaw > 0
        const left = projectModel({ yaw: 20, pitch: 0, roll: 0 });
        expect(left[2][0]).toBeGreaterThan((left[0][0] + left[1][0]) / 2);
        expect(estimateHeadPose(left).yaw).toBeGreaterThan(15);
        // looking up -> nose moves up in the image -> pitch > 0
        const up = projectModel({ yaw: 0, pitch: 15, roll: 0 });
        const front = projectModel({ yaw: 0, pitch: 0, roll: 0 });
        expect(up[2][1] - up[0][1]).toBeLessThan(front[2][1] - front[0][1]);
        expect(estimateHeadPose(up).pitch).toBeGreaterThan(10);
        // clockwise tilt in the image -> image-right eye lower -> roll > 0
        const cw = projectModel({ yaw: 0, pitch: 0, roll: 12 });
        expect(cw[1][1]).toBeGreaterThan(cw[0][1]);
        expect(estimateHeadPose(cw).roll).toBeGreaterThan(9);
    });

    it("is invariant to translation and scale of the keypoints", () => {
        const pts = projectModel({ yaw: 15, pitch: -8, roll: 5 });
        const a = estimateHeadPose(pts);
        const b = estimateHeadPose(pts.map(([x, y]) => [x * 2.5 + 100, y * 2.5 - 40] as Point));
        expect(b.yaw).toBeCloseTo(a.yaw, 6);
        expect(b.pitch).toBeCloseTo(a.pitch, 6);
        expect(b.roll).toBeCloseTo(a.roll, 6);
    });
});
