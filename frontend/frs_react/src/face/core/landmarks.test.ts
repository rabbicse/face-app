import { describe, expect, it } from "vitest";
import { faceGeometryFromMesh, MESH, NormalizedLandmark } from "./landmarks";
import { assessQuality, DEFAULT_QUALITY } from "./quality";

function syntheticMesh(mirror = false): NormalizedLandmark[] {
    const lm: NormalizedLandmark[] = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
    const set = (i: number, x: number, y: number, z = 0) => { lm[i] = { x: mirror ? 1 - x : x, y, z }; };
    set(MESH.eyeA[0], 0.38, 0.40); set(MESH.eyeA[1], 0.46, 0.40);
    set(MESH.eyeB[0], 0.54, 0.40); set(MESH.eyeB[1], 0.62, 0.40);
    set(468, 0.42, 0.40); set(473, 0.58, 0.40);
    set(MESH.noseTip, 0.50, 0.52, -0.05);
    set(MESH.mouthA, 0.43, 0.62); set(MESH.mouthB, 0.57, 0.62);
    set(MESH.forehead, 0.50, 0.20); set(MESH.chin, 0.50, 0.80);
    return lm;
}

describe("face geometry from MediaPipe mesh", () => {
    it("extracts ordered alignment points, box and a frontal pose", () => {
        const g = faceGeometryFromMesh(syntheticMesh(), 640, 480);
        expect(g.points5[0].x).toBeCloseTo(0.42 * 640);
        expect(g.points5[1].x).toBeCloseTo(0.58 * 640);
        expect(g.points5[3].x).toBeLessThan(g.points5[4].x);
        expect(g.interOcular).toBeCloseTo(0.16 * 640);
        expect(Math.abs(g.pose.yaw)).toBeLessThan(1);
        expect(Math.abs(g.pose.roll)).toBeLessThan(1);
        expect(g.box.width).toBeCloseTo(0.24 * 640);
    });

    it("keeps image left/right order for a mirrored frame", () => {
        const g = faceGeometryFromMesh(syntheticMesh(true), 640, 480);
        expect(g.points5[0].x).toBeLessThan(g.points5[1].x);
        expect(g.points5[3].x).toBeLessThan(g.points5[4].x);
    });

    it("needs a full mesh", () => {
        expect(() => faceGeometryFromMesh([], 640, 480)).toThrow();
    });
});

describe("quality gate", () => {
    const good = { box: { x: 200, y: 100, width: 200, height: 250 }, frameWidth: 640, frameHeight: 480, interOcular: 70, sharpness: 150, brightness: 120 };

    it("accepts a good frame", () => {
        const r = assessQuality(good);
        expect(r.ok).toBe(true);
        expect(r.score).toBeGreaterThan(0.7);
    });

    it.each([
        [{ interOcular: 20 }, "too_far"],
        [{ sharpness: 5 }, "blurry"],
        [{ brightness: 20 }, "too_dark"],
        [{ brightness: 250 }, "too_bright"],
        [{ box: { x: 550, y: 100, width: 200, height: 250 } }, "out_of_frame"],
    ])("flags %o as %s", (patch, issue) => {
        const r = assessQuality({ ...good, ...patch }, DEFAULT_QUALITY);
        expect(r.ok).toBe(false);
        expect(r.issues).toContain(issue);
    });
});
