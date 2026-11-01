import type { FaceLandmarker as FaceLandmarkerType, FaceLandmarkerResult } from "@mediapipe/tasks-vision";
import { FaceGeometry, faceGeometryFromMesh } from "../core/landmarks";

export interface LandmarkerOptions {
    modelUrl: string;
    wasmBase: string;
    runningMode: "VIDEO" | "IMAGE";
}

export interface LandmarkerHandle {
    landmarker: FaceLandmarkerType;
    delegate: "GPU" | "CPU";
}

/** MediaPipe Face Landmarker (detector + 478 point mesh with tracking): GPU delegate, CPU fallback. */
export async function createLandmarker(opts: LandmarkerOptions): Promise<LandmarkerHandle> {
    const { FaceLandmarker, FilesetResolver } = await import("@mediapipe/tasks-vision");
    const fileset = await FilesetResolver.forVisionTasks(opts.wasmBase);
    let lastError: unknown;
    for (const delegate of ["GPU", "CPU"] as const) {
        try {
            const landmarker = await FaceLandmarker.createFromOptions(fileset, {
                baseOptions: { modelAssetPath: opts.modelUrl, delegate },
                runningMode: opts.runningMode,
                numFaces: 1,
                minFaceDetectionConfidence: 0.6,
                minFacePresenceConfidence: 0.6,
                minTrackingConfidence: 0.5,
                outputFaceBlendshapes: true, // eyeBlinkLeft / eyeBlinkRight for liveness
                outputFacialTransformationMatrixes: true,
            });
            return { landmarker, delegate };
        } catch (e) {
            lastError = e;
            console.warn(`FaceLandmarker ${delegate} delegate unavailable`, e);
        }
    }
    throw lastError;
}

function toGeometry(result: FaceLandmarkerResult, width: number, height: number): FaceGeometry | null {
    const face = result.faceLandmarks?.[0];
    if (!face) return null;
    const geometry = faceGeometryFromMesh(face, width, height, result.facialTransformationMatrixes?.[0]?.data);
    const shapes = result.faceBlendshapes?.[0]?.categories;
    if (shapes) {
        const score = (name: string) => shapes.find(c => c.categoryName === name)?.score;
        const left = score("eyeBlinkLeft"), right = score("eyeBlinkRight");
        if (left !== undefined && right !== undefined) {
            geometry.eyesClosed = Math.min(left, right);
        }
    }
    return geometry;
}

export function detectInVideo(handle: LandmarkerHandle, video: HTMLVideoElement, timestampMs: number): FaceGeometry | null {
    return toGeometry(handle.landmarker.detectForVideo(video, timestampMs), video.videoWidth, video.videoHeight);
}

export function detectInImage(handle: LandmarkerHandle, image: HTMLImageElement | HTMLCanvasElement | ImageBitmap,
    width: number, height: number): FaceGeometry | null {
    return toGeometry(handle.landmarker.detect(image), width, height);
}
