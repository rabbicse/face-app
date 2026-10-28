/**
 * Per-frame face tracker (MediaPipe Face Landmarker, GPU delegate). Runs every video
 * frame in a few ms and gives: face box, 5 keypoints, head pose and face count.
 * It never produces embeddings — that's the worker's job, and only when needed.
 */
import type { FaceLandmarker as FaceLandmarkerT } from "@mediapipe/tasks-vision";
import { FACE_CONFIG } from "./config";
import { Box, interOcular, Point } from "../core/geometry";
import { landmarksBox, mediapipeTo5, isInsideFrame } from "../core/landmarks";
import { estimateHeadPose, HeadPose } from "../core/pose";

export interface TrackedFace {
    box: Box;
    kps: Point[];
    pose: HeadPose;
    interOcularPx: number;
    inside: boolean;
}

export interface TrackResult {
    faces: TrackedFace[];
    /** main face (largest), if any */
    primary: TrackedFace | null;
    ms: number;
    width: number;
    height: number;
}

let loading: Promise<FaceTracker> | null = null;

export class FaceTracker {
    private lastTs = -1;
    private constructor(private landmarker: FaceLandmarkerT, readonly delegate: "GPU" | "CPU") { }

    static load(): Promise<FaceTracker> {
        if (!loading) loading = FaceTracker.create().catch(e => { loading = null; throw e; });
        return loading;
    }

    private static async create(): Promise<FaceTracker> {
        const { FaceLandmarker, FilesetResolver } = await import("@mediapipe/tasks-vision");
        const fileset = await FilesetResolver.forVisionTasks(FACE_CONFIG.wasm.mediapipe);
        const make = (modelAssetPath: string, delegate: "GPU" | "CPU") => FaceLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath, delegate },
            runningMode: "VIDEO",
            numFaces: 2, // so we can tell the user when a second face is in view
            minFaceDetectionConfidence: 0.5,
            minFacePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: false,
        });
        // self-hosted model first, Google CDN as fallback; GPU first, CPU as fallback
        // (?tracker=cpu / ?tracker=gpu forces a delegate, remembered in localStorage)
        const order = FaceTracker.preferredDelegates();
        for (const url of [FACE_CONFIG.models.landmarker, FACE_CONFIG.models.landmarkerFallback]) {
            for (const delegate of order) {
                try {
                    return new FaceTracker(await make(url, delegate), delegate);
                } catch (e) {
                    console.warn(`[face] landmarker ${delegate} from ${url} failed:`, e);
                }
            }
        }
        throw new Error("Could not load the face landmarker model");
    }

    static preferredDelegates(): ("GPU" | "CPU")[] {
        try {
            const q = new URLSearchParams(location.search).get("tracker")?.toUpperCase();
            if (q === "CPU" || q === "GPU") localStorage.setItem("face.tracker", q);
            if (q === "AUTO") localStorage.removeItem("face.tracker");
            if (localStorage.getItem("face.tracker") === "CPU") return ["CPU", "GPU"];
        } catch { /* storage unavailable */ }
        return ["GPU", "CPU"];
    }

    track(video: HTMLVideoElement, timestampMs: number): TrackResult {
        const t0 = performance.now();
        // MediaPipe requires strictly increasing timestamps
        const ts = timestampMs <= this.lastTs ? this.lastTs + 1 : timestampMs;
        this.lastTs = ts;
        const res = this.landmarker.detectForVideo(video, ts);
        const W = video.videoWidth, H = video.videoHeight;
        const faces: TrackedFace[] = res.faceLandmarks.map(lm => {
            const kps = mediapipeTo5(lm, W, H);
            const box = landmarksBox(lm, W, H);
            return { box, kps, pose: estimateHeadPose(kps), interOcularPx: interOcular(kps), inside: isInsideFrame(box, W, H) };
        });
        faces.sort((a, b) => b.box.width * b.box.height - a.box.width * a.box.height);
        return { faces, primary: faces[0] ?? null, ms: performance.now() - t0, width: W, height: H };
    }

    close() { this.landmarker.close(); loading = null; }
}
