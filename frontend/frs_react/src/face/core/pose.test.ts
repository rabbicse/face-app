import { describe, expect, it } from "vitest";
import { classifyPose, matchesPose, poseFromLandmarks3d, poseFromTransformationMatrix } from "./pose";
import { loadTestVectors } from "./testVectors";

const vectors = loadTestVectors();
const p3 = ([x, y, z]: number[]) => ({ x, y, z });

describe("head pose", () => {
    it.each(vectors.pose.map((c, i) => [i, c] as const))("matches the Python reference (case %i)", (_, c) => {
        const pose = poseFromLandmarks3d(p3(c.points.left_eye), p3(c.points.right_eye), p3(c.points.forehead), p3(c.points.chin));
        expect(pose.yaw).toBeCloseTo(c.pose.yaw, 3);
        expect(pose.pitch).toBeCloseTo(c.pose.pitch, 3);
        expect(pose.roll).toBeCloseTo(c.pose.roll, 3);
        expect(classifyPose(pose)).toBe(c.label);
    });

    it.each(vectors.classify.map(c => [c.pose, c.label] as const))("classifies %o as %s", (pose, label) => {
        expect(classifyPose(pose)).toBe(label);
    });

    it("checks a single target", () => {
        expect(matchesPose({ yaw: 20, pitch: 0, roll: 0 }, "turn_left")).toBe(true);
        expect(matchesPose({ yaw: 20, pitch: 0, roll: 0 }, "turn_right")).toBe(false);
    });
});

/** Column-major 4x4 of R = Rz(roll) * Ry(yaw) * Rx(-pitch) in MediaPipe's metric space (y up, z to camera). */
function transform(yaw: number, pitch: number, roll: number, scale = 1): number[] {
    const [y, p, r] = [yaw, pitch, roll].map(d => (d * Math.PI) / 180);
    const ry = [[Math.cos(y), 0, Math.sin(y)], [0, 1, 0], [-Math.sin(y), 0, Math.cos(y)]];
    const rx = [[1, 0, 0], [0, Math.cos(p), Math.sin(p)], [0, -Math.sin(p), Math.cos(p)]];
    const rz = [[Math.cos(r), -Math.sin(r), 0], [Math.sin(r), Math.cos(r), 0], [0, 0, 1]];
    const mul = (a: number[][], b: number[][]) => a.map(row => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)));
    const m = mul(rz, mul(ry, rx));
    const out: number[] = [];
    for (let col = 0; col < 4; col++) {
        for (let row = 0; row < 4; row++) {
            out.push(row < 3 && col < 3 ? m[row][col] * scale : row === col ? 1 : col === 3 ? [1, -2, -40][row] ?? 1 : 0);
        }
    }
    return out;
}

describe("pose from MediaPipe transformation matrix", () => {
    it.each([
        [0, 0, 0], [25, 0, 0], [-25, 0, 0], [0, 20, 0], [0, -20, 0], [0, 0, 15], [0, 0, -15],
    ])("recovers yaw %d pitch %d roll %d", (yaw, pitch, roll) => {
        const pose = poseFromTransformationMatrix(transform(yaw, pitch, roll, 1.7));
        expect(pose.yaw).toBeCloseTo(yaw, 6);
        expect(pose.pitch).toBeCloseTo(pitch, 6);
        expect(pose.roll).toBeCloseTo(roll, 6);
    });

    it("is preferred over the landmark estimate", async () => {
        const { faceGeometryFromMesh } = await import("./landmarks");
        const lm = Array.from({ length: 478 }, (_, i) => ({ x: 0.3 + (i % 20) / 50, y: 0.3 + Math.floor(i / 20) / 60, z: 0 }));
        const g = faceGeometryFromMesh(lm, 640, 480, transform(-25, 0, 0));
        expect(g.pose.yaw).toBeCloseTo(-25, 6);
    });
});
