import { describe, expect, it } from "vitest";
import { headAxes } from "./overlay";

const close = (a: number[], b: number[]) => a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 6));

describe("headAxes", () => {
    it("is the identity for a frontal pose: nose (z) towards the camera", () => {
        const { x, y, z } = headAxes({ yaw: 0, pitch: 0, roll: 0 });
        close(x, [1, 0, 0]);
        close(y, [0, 1, 0]);
        close(z, [0, 0, 1]);
    });

    it("follows the shared head-pose convention", () => {
        // yaw > 0: face turned to the image's right
        expect(headAxes({ yaw: 30, pitch: 0, roll: 0 }).z[0]).toBeGreaterThan(0.4);
        // pitch > 0: looking up
        expect(headAxes({ yaw: 0, pitch: 30, roll: 0 }).z[1]).toBeGreaterThan(0.4);
        // roll > 0: counter-clockwise, so the head's x axis tips upwards
        expect(headAxes({ yaw: 0, pitch: 0, roll: 30 }).x[1]).toBeGreaterThan(0.4);
    });

    it("keeps the axes orthonormal", () => {
        const { x, y, z } = headAxes({ yaw: 25, pitch: -12, roll: 8 });
        const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);
        expect(dot(x, x)).toBeCloseTo(1, 6);
        expect(dot(z, z)).toBeCloseTo(1, 6);
        expect(dot(x, y)).toBeCloseTo(0, 6);
        expect(dot(y, z)).toBeCloseTo(0, 6);
    });
});
