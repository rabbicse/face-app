import { describe, expect, it } from "vitest";
import { classifyPose, matchesPose, poseFromLandmarks3d } from "./pose";
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
