import { dot, fuseEmbeddings } from "./embedding";
import { HeadPose, matchesPose, PoseLabel } from "./pose";

export interface PoseRequirement {
    label: PoseLabel;
    count: number;
    instruction: string;
}

/** Instructions are phrased for the user looking at a mirrored (selfie) preview. */
export const FULL_REGISTRATION_PLAN: PoseRequirement[] = [
    { label: "frontal", count: 3, instruction: "Look straight at the camera" },
    { label: "turn_left", count: 2, instruction: "Slowly turn your head to your left" },
    { label: "turn_right", count: 2, instruction: "Slowly turn your head to your right" },
    { label: "look_up", count: 1, instruction: "Tilt your head slightly up" },
    { label: "look_down", count: 1, instruction: "Tilt your head slightly down" },
];

export const QUICK_REGISTRATION_PLAN: PoseRequirement[] = [
    { label: "frontal", count: 3, instruction: "Look straight at the camera" },
];

export interface RegistrationSample {
    embedding: Float32Array;
    pose: HeadPose;
    poseLabel: PoseLabel;
    quality: number;
    timestamp: number;
}

export type OfferResult = "accepted" | "complete" | "wrong_pose" | "too_soon" | "inconsistent" | "duplicate";

export interface RegistrationOptions {
    /** minimum time between two accepted samples */
    minIntervalMs: number;
    /** min cosine to the frontal template for non frontal samples (pose changes lower the similarity) */
    minCrossPoseSimilarity: number;
    /** min cosine to the already collected samples of the same pose */
    minSamePoseSimilarity: number;
    /** samples of the same pose more similar than this add no information */
    maxSamePoseSimilarity: number;
}

export const DEFAULT_REGISTRATION_OPTIONS: RegistrationOptions = {
    minIntervalMs: 300,
    minCrossPoseSimilarity: 0.3,
    minSamePoseSimilarity: 0.5,
    maxSamePoseSimilarity: 0.995,
};

export interface PoseProgress {
    label: PoseLabel;
    have: number;
    need: number;
}

/**
 * Guided multi pose enrolment. Poses are collected in plan order; a sample is only accepted when the
 * head pose matches the current target and the embedding is consistent with what was collected so far
 * (protects against a different person stepping in during the capture).
 */
export class RegistrationSession {
    private readonly samples: RegistrationSample[] = [];
    private lastAccepted = -Infinity;

    constructor(readonly plan: readonly PoseRequirement[] = FULL_REGISTRATION_PLAN,
        private readonly options: RegistrationOptions = DEFAULT_REGISTRATION_OPTIONS) {
        if (plan.length === 0) {
            throw new Error("registration plan is empty");
        }
    }

    get currentTarget(): PoseRequirement | null {
        return this.plan.find(r => this.count(r.label) < r.count) ?? null;
    }

    get isComplete(): boolean {
        return this.currentTarget === null;
    }

    get progress(): PoseProgress[] {
        return this.plan.map(r => ({ label: r.label, have: this.count(r.label), need: r.count }));
    }

    /** 0..1 overall progress */
    get fraction(): number {
        const need = this.plan.reduce((s, r) => s + r.count, 0);
        const have = this.plan.reduce((s, r) => s + Math.min(r.count, this.count(r.label)), 0);
        return have / need;
    }

    get collected(): readonly RegistrationSample[] {
        return this.samples;
    }

    /**
     * Pose relative to the user's own frontal baseline (mean of the accepted frontal samples). Non frontal
     * targets are checked against it, so a camera mounted below / above the eyes does not bias them.
     */
    relativePose(pose: HeadPose): HeadPose {
        const frontal = this.samples.filter(s => s.poseLabel === "frontal");
        if (this.currentTarget?.label === "frontal" || frontal.length === 0) {
            return pose;
        }
        const yaw0 = frontal.reduce((s, f) => s + f.pose.yaw, 0) / frontal.length;
        const pitch0 = frontal.reduce((s, f) => s + f.pose.pitch, 0) / frontal.length;
        return { yaw: pose.yaw - yaw0, pitch: pose.pitch - pitch0, roll: pose.roll };
    }

    /** Cheap pre-check so callers only compute an embedding when it could be accepted. */
    wants(pose: HeadPose, timestamp: number): boolean {
        const target = this.currentTarget;
        return target !== null && timestamp - this.lastAccepted >= this.options.minIntervalMs
            && matchesPose(this.relativePose(pose), target.label);
    }

    offer(sample: Omit<RegistrationSample, "poseLabel">): OfferResult {
        const target = this.currentTarget;
        if (target === null) return "complete";
        if (sample.timestamp - this.lastAccepted < this.options.minIntervalMs) return "too_soon";
        if (!matchesPose(this.relativePose(sample.pose), target.label)) return "wrong_pose";

        const samePose = this.samples.filter(s => s.poseLabel === target.label);
        for (const s of samePose) {
            const sim = dot(s.embedding, sample.embedding);
            if (sim < this.options.minSamePoseSimilarity) return "inconsistent";
            if (sim > this.options.maxSamePoseSimilarity) return "duplicate";
        }
        const frontal = this.samples.filter(s => s.poseLabel === "frontal");
        if (target.label !== "frontal" && frontal.length > 0) {
            const template = fuseEmbeddings(frontal.map(s => s.embedding));
            if (dot(template, sample.embedding) < this.options.minCrossPoseSimilarity) return "inconsistent";
        }

        this.samples.push({ ...sample, poseLabel: target.label });
        this.lastAccepted = sample.timestamp;
        return this.isComplete ? "complete" : "accepted";
    }

    reset(): void {
        this.samples.length = 0;
        this.lastAccepted = -Infinity;
    }

    private count(label: PoseLabel): number {
        let n = 0;
        for (const s of this.samples) {
            if (s.poseLabel === label) n++;
        }
        return n;
    }
}
