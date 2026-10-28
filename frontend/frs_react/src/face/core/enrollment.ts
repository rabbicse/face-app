/**
 * Guided multi-pose enrollment (registration).
 *
 * Registration uses yaw / pitch / roll: the user is asked to look straight, then turn a
 * little left / right / up / down. One embedding per pose gives a gallery that matches
 * the person later from natural, non-frontal angles — the main accuracy win over a
 * single frontal template. Identification never asks for a pose.
 *
 * The session is a pure state machine (no timers, no DOM) so it is unit-tested and the
 * same logic is ported to Dart for the Flutter app.
 */
import { HeadPose } from "./pose";
import { FaceQuality, qualityScore } from "./quality";
import { cosine, meanEmbedding } from "./vector";

export type PoseStepId = "front" | "left" | "right" | "up" | "down";

export interface PoseStep {
    id: PoseStepId;
    instruction: string;
    yaw: [number, number];
    pitch: [number, number];
}

/** Ranges in degrees; see pose.ts for sign conventions. */
export const DEFAULT_POSE_STEPS: PoseStep[] = [
    { id: "front", instruction: "Look straight at the camera", yaw: [-8, 8], pitch: [-8, 8] },
    { id: "left", instruction: "Slowly turn your head to your left", yaw: [14, 35], pitch: [-12, 12] },
    { id: "right", instruction: "Slowly turn your head to your right", yaw: [-35, -14], pitch: [-12, 12] },
    { id: "up", instruction: "Tilt your chin up a little", yaw: [-12, 12], pitch: [9, 28] },
    { id: "down", instruction: "Tilt your chin down a little", yaw: [-12, 12], pitch: [-28, -9] },
];

export interface EnrollmentConfig {
    steps: PoseStep[];
    /** consecutive in-range tracker frames before we start sampling (avoids motion blur) */
    holdFrames: number;
    /** good embeddings collected per step; the best one is kept */
    samplesPerStep: number;
    maxAbsRoll: number;
    /** every sample must match the frontal template at least this much (same person check) */
    minSelfSimilarity: number;
}

export const DEFAULT_ENROLLMENT_CONFIG: EnrollmentConfig = {
    steps: DEFAULT_POSE_STEPS,
    holdFrames: 4,
    samplesPerStep: 3,
    maxAbsRoll: 15,
    minSelfSimilarity: 0.5,
};

export interface EnrollmentSample {
    step: PoseStepId;
    embedding: Float32Array;
    pose: HeadPose;
    quality: FaceQuality;
}

export interface TrackerFrame {
    /** pose from the per-frame tracker, null when no face */
    pose: HeadPose | null;
    /** false when the tracker sees several faces / face out of frame etc. */
    faceOk: boolean;
}

export type Guidance =
    | { kind: "no_face"; message: string }
    | { kind: "adjust"; message: string; arrow: "left" | "right" | "up" | "down" | "straighten" | null }
    | { kind: "hold"; message: string }
    | { kind: "capturing"; message: string }
    | { kind: "done"; message: string };

export interface EnrollmentProgress {
    stepIndex: number;
    step: PoseStep | null;
    completedSteps: PoseStepId[];
    samplesInStep: number;
    fraction: number;
}

export type SampleVerdict = "accepted" | "rejected_pose" | "rejected_identity" | "ignored";

export class EnrollmentSession {
    readonly config: EnrollmentConfig;
    private stepIndex = 0;
    private holdCount = 0;
    private readonly samples = new Map<PoseStepId, EnrollmentSample[]>();

    constructor(config: Partial<EnrollmentConfig> = {}) {
        this.config = { ...DEFAULT_ENROLLMENT_CONFIG, ...config };
    }

    get currentStep(): PoseStep | null {
        return this.config.steps[this.stepIndex] ?? null;
    }

    get isComplete(): boolean {
        return this.stepIndex >= this.config.steps.length;
    }

    poseMatches(step: PoseStep, pose: HeadPose): boolean {
        return pose.yaw >= step.yaw[0] && pose.yaw <= step.yaw[1]
            && pose.pitch >= step.pitch[0] && pose.pitch <= step.pitch[1]
            && Math.abs(pose.roll) <= this.config.maxAbsRoll;
    }

    /**
     * Call on every tracker frame (cheap). Returns guidance for the UI and whether an
     * embedding should be computed for this frame.
     */
    onFrame(frame: TrackerFrame): { guidance: Guidance; wantEmbedding: boolean } {
        const step = this.currentStep;
        if (!step) return { guidance: { kind: "done", message: "All set!" }, wantEmbedding: false };
        if (!frame.pose || !frame.faceOk) {
            this.holdCount = 0;
            return { guidance: { kind: "no_face", message: "Position your face inside the circle" }, wantEmbedding: false };
        }
        const p = frame.pose;
        if (!this.poseMatches(step, p)) {
            this.holdCount = 0;
            return { guidance: this.adjustGuidance(step, p), wantEmbedding: false };
        }
        this.holdCount++;
        if (this.holdCount < this.config.holdFrames) {
            return { guidance: { kind: "hold", message: "Hold it there…" }, wantEmbedding: false };
        }
        return { guidance: { kind: "capturing", message: "Capturing…" }, wantEmbedding: true };
    }

    private adjustGuidance(step: PoseStep, p: HeadPose): Guidance {
        if (Math.abs(p.roll) > this.config.maxAbsRoll) {
            return { kind: "adjust", message: "Keep your head straight (don't tilt sideways)", arrow: "straighten" };
        }
        const [y0, y1] = step.yaw, [p0, p1] = step.pitch;
        // largest violation first
        const dy = p.yaw < y0 ? y0 - p.yaw : p.yaw > y1 ? p.yaw - y1 : 0;
        const dp = p.pitch < p0 ? p0 - p.pitch : p.pitch > p1 ? p.pitch - p1 : 0;
        if (dy >= dp) {
            return p.yaw < y0
                ? { kind: "adjust", message: step.id === "left" ? step.instruction : "Turn slightly to your left", arrow: "left" }
                : { kind: "adjust", message: step.id === "right" ? step.instruction : "Turn slightly to your right", arrow: "right" };
        }
        return p.pitch < p0
            ? { kind: "adjust", message: step.id === "up" ? step.instruction : "Raise your chin slightly", arrow: "up" }
            : { kind: "adjust", message: step.id === "down" ? step.instruction : "Lower your chin slightly", arrow: "down" };
    }

    /**
     * Submit an embedding computed for a frame where onFrame() asked for one.
     * `pose` must be the pose at capture time; the sample is re-checked against the step.
     */
    addSample(sample: Omit<EnrollmentSample, "step">): SampleVerdict {
        const step = this.currentStep;
        if (!step) return "ignored";
        if (!this.poseMatches(step, sample.pose)) return "rejected_pose";

        const front = this.samples.get("front");
        if (step.id !== "front" && front && front.length > 0) {
            const ref = meanEmbedding(front.map(s => s.embedding));
            if (cosine(ref, sample.embedding) < this.config.minSelfSimilarity) return "rejected_identity";
        }
        if (step.id === "front") {
            // samples within the frontal step must agree with each other
            const prev = this.samples.get("front") ?? [];
            if (prev.some(s => cosine(s.embedding, sample.embedding) < this.config.minSelfSimilarity)) {
                this.samples.set("front", []);
                this.holdCount = 0;
                return "rejected_identity";
            }
        }
        const list = this.samples.get(step.id) ?? [];
        list.push({ ...sample, step: step.id });
        this.samples.set(step.id, list);
        if (list.length >= this.config.samplesPerStep) {
            this.stepIndex++;
            this.holdCount = 0;
        }
        return "accepted";
    }

    /** Lets the user skip a step that their camera / mobility can't reach (not "front"). */
    skipStep(): boolean {
        const step = this.currentStep;
        if (!step || step.id === "front") return false;
        this.samples.delete(step.id);
        this.stepIndex++;
        this.holdCount = 0;
        return true;
    }

    reset(): void {
        this.stepIndex = 0;
        this.holdCount = 0;
        this.samples.clear();
    }

    progress(): EnrollmentProgress {
        const steps = this.config.steps;
        const inStep = this.currentStep ? (this.samples.get(this.currentStep.id)?.length ?? 0) : 0;
        return {
            stepIndex: this.stepIndex,
            step: this.currentStep,
            completedSteps: steps.slice(0, this.stepIndex).map(s => s.id),
            samplesInStep: inStep,
            fraction: Math.min(1, (this.stepIndex + inStep / this.config.samplesPerStep) / steps.length),
        };
    }

    /**
     * Final templates: the best-quality sample of each completed pose step plus a
     * quality-weighted centroid of all frontal samples.
     */
    result(): { templates: EnrollmentSample[]; centroid: Float32Array } {
        if (!this.samples.get("front")?.length) throw new Error("enrollment has no frontal sample");
        const templates: EnrollmentSample[] = [];
        for (const step of this.config.steps) {
            const list = this.samples.get(step.id);
            if (!list?.length) continue;
            templates.push(list.reduce((a, b) => (qualityScore(b.quality) > qualityScore(a.quality) ? b : a)));
        }
        const all = [...this.samples.values()].flat();
        const centroid = meanEmbedding(all.map(s => s.embedding), all.map(s => qualityScore(s.quality)));
        return { templates, centroid };
    }
}
