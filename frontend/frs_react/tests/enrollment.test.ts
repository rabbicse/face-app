import { describe, expect, it } from "vitest";
import { EnrollmentSession, DEFAULT_POSE_STEPS } from "@/face/core/enrollment";
import { FaceQuality } from "@/face/core/quality";
import { l2Normalize, dot } from "@/face/core/vector";
import { HeadPose } from "@/face/core/pose";

const Q: FaceQuality = { sharpness: 200, brightness: 130, contrast: 40, interOcularPx: 80, detScore: 0.9 };

function personVector(seed: number, noise = 0, salt = 0): Float32Array {
    const v = new Float32Array(512);
    let s = seed;
    for (let i = 0; i < 512; i++) { s = (s * 16807) % 2147483647; v[i] = (s / 2147483647) - 0.5; }
    let t = 99 + salt;
    for (let i = 0; i < 512; i++) { t = (t * 48271) % 2147483647; v[i] += noise * ((t / 2147483647) - 0.5); }
    return l2Normalize(v);
}

const POSE_FOR: Record<string, HeadPose> = {
    front: { yaw: 0, pitch: 0, roll: 0 },
    left: { yaw: 22, pitch: 0, roll: 2 },
    right: { yaw: -22, pitch: 1, roll: -2 },
    up: { yaw: 1, pitch: 16, roll: 0 },
    down: { yaw: -1, pitch: -16, roll: 0 },
};

/** drives the session like the camera loop does */
function capture(s: EnrollmentSession, pose: HeadPose, embedding: Float32Array) {
    let r = s.onFrame({ pose, faceOk: true });
    let guard = 0;
    while (!r.wantEmbedding && guard++ < 20) r = s.onFrame({ pose, faceOk: true });
    return s.addSample({ embedding, pose, quality: Q });
}

describe("EnrollmentSession", () => {
    it("walks through all pose steps and returns one template per pose", () => {
        const s = new EnrollmentSession();
        let salt = 0;
        for (const step of DEFAULT_POSE_STEPS) {
            for (let k = 0; k < s.config.samplesPerStep; k++) {
                expect(s.currentStep?.id).toBe(step.id);
                expect(capture(s, POSE_FOR[step.id], personVector(1, 0.3, salt++))).toBe("accepted");
            }
        }
        expect(s.isComplete).toBe(true);
        const { templates, centroid } = s.result();
        expect(templates.map(t => t.step)).toEqual(["front", "left", "right", "up", "down"]);
        expect(dot(centroid, personVector(1))).toBeGreaterThan(0.9);
        expect(s.progress().fraction).toBe(1);
    });

    it("requires the pose to be held before sampling", () => {
        const s = new EnrollmentSession({ holdFrames: 3 });
        const f = { pose: POSE_FOR.front, faceOk: true };
        expect(s.onFrame(f).wantEmbedding).toBe(false);
        expect(s.onFrame(f).guidance.kind).toBe("hold");
        expect(s.onFrame(f).wantEmbedding).toBe(true);
        // losing the face resets the hold counter
        s.onFrame({ pose: null, faceOk: false });
        expect(s.onFrame(f).wantEmbedding).toBe(false);
    });

    it("gives directional guidance", () => {
        const s = new EnrollmentSession();
        // finish front
        for (let k = 0; k < 3; k++) capture(s, POSE_FOR.front, personVector(1, 0.2, k));
        expect(s.currentStep?.id).toBe("left");
        const g = s.onFrame({ pose: POSE_FOR.front, faceOk: true }).guidance;
        expect(g.kind).toBe("adjust");
        expect(g.kind === "adjust" && g.arrow).toBe("left");
        const tilted = s.onFrame({ pose: { yaw: 22, pitch: 0, roll: 25 }, faceOk: true }).guidance;
        expect(tilted.kind === "adjust" && tilted.arrow).toBe("straighten");
        expect(s.onFrame({ pose: null, faceOk: false }).guidance.kind).toBe("no_face");
    });

    it("rejects a sample whose pose drifted out of range at capture time", () => {
        const s = new EnrollmentSession();
        s.onFrame({ pose: POSE_FOR.front, faceOk: true });
        expect(s.addSample({ embedding: personVector(1), pose: POSE_FOR.left, quality: Q })).toBe("rejected_pose");
    });

    it("rejects a different person appearing mid-enrollment", () => {
        const s = new EnrollmentSession();
        for (let k = 0; k < 3; k++) capture(s, POSE_FOR.front, personVector(1, 0.2, k));
        expect(capture(s, POSE_FOR.left, personVector(2))).toBe("rejected_identity");
        expect(capture(s, POSE_FOR.left, personVector(1, 0.2, 9))).toBe("accepted");
    });

    it("allows skipping non-frontal steps but never the frontal one", () => {
        const s = new EnrollmentSession();
        expect(s.skipStep()).toBe(false);
        for (let k = 0; k < 3; k++) capture(s, POSE_FOR.front, personVector(1, 0.2, k));
        expect(s.skipStep()).toBe(true);
        expect(s.currentStep?.id).toBe("right");
    });

    it("refuses to produce a result without a frontal template", () => {
        expect(() => new EnrollmentSession().result()).toThrow();
    });
});
