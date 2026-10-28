/**
 * FaceCaptureController: the real-time loop, deliberately outside React.
 *
 *  camera frame ──► tracker (MediaPipe, ~3-8 ms, every new frame)
 *        │                └─► overlay + mode.onTrack()  → guidance (UI store, only on change)
 *        └─ if mode wants an embedding and the worker is idle:
 *             square ROI (192 px) ──transfer──► worker: SCRFD kps + align + ArcFace 512-d
 *                                                   └─► mode.onEmbedding()
 *
 * Old pipeline for comparison: detection + JPEG encode + HTTP upload + server-side
 * RetinaFace R50 + ArcFace R100 on CPU, every frame, awaited inside requestAnimationFrame.
 */
import { FACE_CONFIG } from "./config";
import { EmbedderClient } from "./embedderClient";
import { FaceTracker, TrackResult, TrackedFace } from "./tracker";
import { frameLoop } from "./camera";
import { createStore, Store } from "./store";
import { Point, squareRoi } from "../core/geometry";
import type { WorkerEmbedding } from "./protocol";

export interface CaptureStats {
    fps: number;
    trackMs: number;
    detectMs: number;
    embedMs: number;
    roundTripMs: number;
    backend: string;
    delegate: string;
}

export interface CaptureState {
    status: "idle" | "running" | "paused" | "done" | "error";
    message: string;
    arrow: "left" | "right" | "up" | "down" | "straighten" | null;
    tone: "info" | "good" | "warn" | "error";
    progress: number;
    faceVisible: boolean;
    stats: CaptureStats;
}

export interface ModeContext {
    state: Store<CaptureState>;
    now: number;
    pause(): void;
}

export interface CaptureMode {
    /** called every frame; return true to request an embedding for this frame */
    onTrack(track: TrackResult, ctx: ModeContext): boolean;
    /** called when an embedding requested for `face` is ready (may be async) */
    onEmbedding(result: WorkerEmbedding | null, face: TrackedFace, ctx: ModeContext): void | Promise<void>;
}

export interface ControllerOptions {
    drawLandmarks?: boolean;
    wantAligned?: boolean;
}

export class FaceCaptureController {
    readonly state: Store<CaptureState>;
    private stopLoop: (() => void) | null = null;
    private roiCanvas: OffscreenCanvas | HTMLCanvasElement;
    private paused = false;
    private inflight = false;
    private lastFrameTs = 0;
    private fps = 0;
    private lastStatsPublish = 0;
    private stats: CaptureStats = { fps: 0, trackMs: 0, detectMs: 0, embedMs: 0, roundTripMs: 0, backend: "", delegate: "" };
    lastAligned: ImageData | null = null;

    constructor(
        private video: HTMLVideoElement,
        private overlay: HTMLCanvasElement | null,
        private tracker: FaceTracker,
        private embedder: EmbedderClient,
        private mode: CaptureMode,
        private opts: ControllerOptions = {},
    ) {
        const S = FACE_CONFIG.roiSize;
        this.roiCanvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(S, S) : Object.assign(document.createElement("canvas"), { width: S, height: S });
        this.state = createStore<CaptureState>({
            status: "idle", message: "Starting…", arrow: null, tone: "info", progress: 0, faceVisible: false, stats: this.stats,
        });
        this.stats.delegate = tracker.delegate;
        this.stats.backend = embedder.info?.backend ?? "";
    }

    start(): void {
        if (this.stopLoop) return;
        this.paused = false;
        if (this.overlay) {
            // size the overlay ONCE (the old code resized it every frame, which also cleared it)
            this.overlay.width = this.video.videoWidth;
            this.overlay.height = this.video.videoHeight;
        }
        this.state.set({ status: "running" });
        this.stopLoop = frameLoop(this.video, ts => this.onFrame(ts));
    }

    stop(): void {
        this.stopLoop?.();
        this.stopLoop = null;
    }

    pause(): void { this.paused = true; this.state.set({ status: "paused" }); }
    resume(): void { this.paused = false; this.state.set({ status: "running" }); }

    private ctx(now: number): ModeContext {
        return { state: this.state, now, pause: () => this.pause() };
    }

    private onFrame(ts: number): void {
        if (this.paused || this.video.readyState < 2) return;
        if (this.lastFrameTs) {
            const inst = 1000 / Math.max(1, ts - this.lastFrameTs);
            this.fps = this.fps ? this.fps * 0.9 + inst * 0.1 : inst;
        }
        this.lastFrameTs = ts;

        const track = this.tracker.track(this.video, ts);
        this.stats.trackMs = this.stats.trackMs * 0.9 + track.ms * 0.1;
        this.drawOverlay(track);
        this.state.set({ faceVisible: !!track.primary });

        const want = this.mode.onTrack(track, this.ctx(ts));
        if (want && track.primary && !this.inflight && !this.embedder.isBusy) {
            void this.requestEmbedding(track.primary); // fire & forget; frames keep flowing
        }
        if (ts - this.lastStatsPublish > 500) {
            this.lastStatsPublish = ts;
            this.stats.fps = this.fps;
            this.stats.backend = this.embedder.info?.backend ?? this.stats.backend;
            this.state.set({ stats: { ...this.stats } });
        }
    }

    /** Crops a square ROI around the face on the GPU and ships it to the worker. */
    private async requestEmbedding(face: TrackedFace): Promise<void> {
        this.inflight = true;
        try {
            await this.embedFace(face);
        } finally {
            this.inflight = false;
        }
    }

    private async embedFace(face: TrackedFace): Promise<void> {
        const S = FACE_CONFIG.roiSize;
        const roi = squareRoi(face.box, FACE_CONFIG.roiScale);
        const k = S / roi.side;
        const g = this.roiCanvas.getContext("2d") as OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
        g.fillStyle = "#000";
        g.fillRect(0, 0, S, S);
        // clip the source rect to the frame ourselves (consistent across browsers)
        const W = this.video.videoWidth, H = this.video.videoHeight;
        const sx = Math.max(0, roi.x), sy = Math.max(0, roi.y);
        const ex = Math.min(W, roi.x + roi.side), ey = Math.min(H, roi.y + roi.side);
        if (ex <= sx || ey <= sy) return;
        g.drawImage(this.video, sx, sy, ex - sx, ey - sy, (sx - roi.x) * k, (sy - roi.y) * k, (ex - sx) * k, (ey - sy) * k);
        const bitmap = this.roiCanvas instanceof HTMLCanvasElement
            ? await createImageBitmap(this.roiCanvas)
            : this.roiCanvas.transferToImageBitmap();
        const fallback = face.kps.map(([x, y]) => [(x - roi.x) * k, (y - roi.y) * k] as Point);

        const t0 = performance.now();
        const result = await this.embedder.embedRoi(bitmap, 1 / k, fallback, this.opts.wantAligned);
        if (result === "busy") return;
        this.stats.roundTripMs = performance.now() - t0;
        if (result) {
            this.stats.detectMs = result.timings.detectMs;
            this.stats.embedMs = result.timings.embedMs;
            if (result.aligned) this.lastAligned = result.aligned;
        }
        if (this.paused || !this.stopLoop) return;
        await this.mode.onEmbedding(result, face, this.ctx(performance.now()));
    }

    private drawOverlay(track: TrackResult): void {
        const c = this.overlay;
        if (!c) return;
        const g = c.getContext("2d")!;
        g.clearRect(0, 0, c.width, c.height);
        if (!this.opts.drawLandmarks) return;
        for (const f of track.faces) {
            g.strokeStyle = f === track.primary ? "rgba(34,197,94,0.9)" : "rgba(239,68,68,0.9)";
            g.lineWidth = 2;
            g.strokeRect(f.box.x, f.box.y, f.box.width, f.box.height);
            g.fillStyle = "rgba(250,204,21,0.95)";
            for (const [x, y] of f.kps) { g.beginPath(); g.arc(x, y, 3, 0, Math.PI * 2); g.fill(); }
        }
    }
}
