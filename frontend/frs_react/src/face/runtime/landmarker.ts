import type { FaceLandmarker as FaceLandmarkerType } from "@mediapipe/tasks-vision";
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
                outputFaceBlendshapes: false,
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

export function detectInVideo(handle: LandmarkerHandle, video: HTMLVideoElement, timestampMs: number): FaceGeometry | null {
    const result = handle.landmarker.detectForVideo(video, timestampMs);
    const face = result.faceLandmarks?.[0];
    return face ? faceGeometryFromMesh(face, video.videoWidth, video.videoHeight, result.facialTransformationMatrixes?.[0]?.data) : null;
}

export function detectInImage(handle: LandmarkerHandle, image: HTMLImageElement | HTMLCanvasElement | ImageBitmap,
    width: number, height: number): FaceGeometry | null {
    const result = handle.landmarker.detect(image);
    const face = result.faceLandmarks?.[0];
    return face ? faceGeometryFromMesh(face, width, height, result.facialTransformationMatrixes?.[0]?.data) : null;
}
