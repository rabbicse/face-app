/** Enrollment (multi-pose) and identification (pose-free) capture modes. */
import { CaptureMode, ModeContext } from "./controller";
import type { TrackResult, TrackedFace } from "./tracker";
import type { WorkerEmbedding } from "./protocol";
import { DEFAULT_POSE_STEPS, EnrollmentSession, EnrollmentSample, PoseStepId } from "../core/enrollment";
import { ENROLL_QUALITY, IDENTIFY_QUALITY, QUALITY_MESSAGES, qualityIssues } from "../core/quality";
import { FrameFusion } from "../core/identification";
import type { IdentifyResponse } from "../repository";
import { FACE_CONFIG } from "./config";

function trackerGate(track: TrackResult, minIod: number): string | null {
    if (!track.primary) return "Position your face inside the circle";
    if (track.faces.length > 1 && track.faces[1].box.width > track.primary.box.width * 0.6) return "Only one person in front of the camera, please";
    if (!track.primary.inside) return "Move your face fully into the frame";
    if (track.primary.interOcularPx < minIod) return "Move a bit closer to the camera";
    return null;
}

// ---------------------------------------------------------------------------------------
export class EnrollmentMode implements CaptureMode {
    readonly session: EnrollmentSession;
    private lastIssueAt = 0;

    constructor(private onComplete: (templates: EnrollmentSample[]) => void, steps?: PoseStepId[]) {
        const chosen = steps?.length ? DEFAULT_POSE_STEPS.filter(s => s.id === "front" || steps.includes(s.id)) : DEFAULT_POSE_STEPS;
        this.session = new EnrollmentSession({ steps: chosen });
    }

    onTrack(track: TrackResult, ctx: ModeContext): boolean {
        const gate = trackerGate(track, ENROLL_QUALITY.minInterOcularPx * 0.9);
        const { guidance, wantEmbedding } = this.session.onFrame({ pose: track.primary?.pose ?? null, faceOk: gate === null });
        // keep a quality warning on screen briefly instead of flickering
        if (ctx.now - this.lastIssueAt > 900) {
            ctx.state.set({
                message: gate ?? guidance.message,
                arrow: gate ? null : guidance.kind === "adjust" ? guidance.arrow : null,
                tone: gate ? "warn" : guidance.kind === "capturing" || guidance.kind === "hold" ? "good" : "info",
                progress: this.session.progress().fraction,
            });
        }
        return wantEmbedding;
    }

    onEmbedding(r: WorkerEmbedding | null, face: TrackedFace, ctx: ModeContext): void {
        if (!r) return;
        const issues = qualityIssues(r.quality, ENROLL_QUALITY);
        if (issues.length) {
            this.lastIssueAt = ctx.now;
            ctx.state.set({ message: QUALITY_MESSAGES[issues[0]], tone: "warn", arrow: null });
            return;
        }
        const verdict = this.session.addSample({ embedding: r.embedding, pose: face.pose, quality: r.quality });
        if (verdict === "rejected_identity") {
            this.lastIssueAt = ctx.now;
            ctx.state.set({ message: "A different face was detected — please stay in front of the camera", tone: "error" });
        }
        ctx.state.set({ progress: this.session.progress().fraction });
        if (this.session.isComplete) {
            ctx.pause();
            ctx.state.set({ status: "done", message: "Face captured from all angles", tone: "good", arrow: null, progress: 1 });
            this.onComplete(this.session.result().templates);
        }
    }

    skip(): boolean { return this.session.skipStep(); }
}

// ---------------------------------------------------------------------------------------
export class IdentifyMode implements CaptureMode {
    private fusion = new FrameFusion();
    private lastJob = 0;
    private querying = false;
    private lastIssueAt = 0;
    attempts = 0;

    constructor(
        private query: (embedding: Float32Array) => Promise<IdentifyResponse>,
        private onResult: (r: IdentifyResponse) => "stop" | "continue",
    ) { }

    onTrack(track: TrackResult, ctx: ModeContext): boolean {
        if (this.querying) return false;
        const gate = trackerGate(track, IDENTIFY_QUALITY.minInterOcularPx);
        if (gate) {
            if (!track.primary) this.fusion.reset();
            if (ctx.now - this.lastIssueAt > 900) ctx.state.set({ message: gate, tone: "warn", arrow: null });
            return false;
        }
        if (ctx.now - this.lastIssueAt > 900) {
            ctx.state.set({ message: "Hold still…", tone: "info", progress: this.fusion.count / this.fusion.config.frames });
        }
        if (ctx.now - this.lastJob < FACE_CONFIG.identifyIntervalMs) return false;
        this.lastJob = ctx.now;
        return true;
    }

    async onEmbedding(r: WorkerEmbedding | null, _face: TrackedFace, ctx: ModeContext): Promise<void> {
        if (!r || this.querying) return;
        const issues = qualityIssues(r.quality, IDENTIFY_QUALITY);
        if (issues.length) {
            this.lastIssueAt = ctx.now;
            ctx.state.set({ message: QUALITY_MESSAGES[issues[0]], tone: "warn" });
            return;
        }
        this.fusion.push(r.embedding, r.quality, ctx.now);
        ctx.state.set({ progress: this.fusion.count / this.fusion.config.frames });
        if (!this.fusion.ready()) return;

        // one fused query per attempt (not one HTTP request per video frame)
        this.querying = true;
        ctx.state.set({ message: "Matching…", tone: "info" });
        try {
            const res = await this.query(this.fusion.fused());
            this.attempts++;
            if (this.onResult(res) === "stop") {
                ctx.pause();
                return;
            }
            this.lastIssueAt = performance.now();
            ctx.state.set({ message: "Face not recognised — try again", tone: "error", progress: 0 });
        } catch (e) {
            this.lastIssueAt = performance.now() + 3000;
            ctx.state.set({ message: e instanceof Error ? e.message : String(e), tone: "error" });
        } finally {
            this.fusion.reset();
            this.querying = false;
        }
    }
}
