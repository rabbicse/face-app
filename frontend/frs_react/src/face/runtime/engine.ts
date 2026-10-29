import { alignmentMatrix } from "../core/alignment";
import { FaceGeometry } from "../core/landmarks";
import { laplacianVariance, meanValue, toGray } from "../core/preprocess";
import { assessQuality, QualityReport, QualityThresholds, DEFAULT_QUALITY } from "../core/quality";
import { FACE_CONFIG } from "../config";
import { EmbedderClient, EmbeddingResult } from "./embedderClient";
import { createLandmarker, detectInImage, detectInVideo, LandmarkerHandle } from "./landmarker";
import { createCropContext, warpFace } from "./warp";

export interface AlignedFace {
    geometry: FaceGeometry;
    crop: ImageData;
    quality: QualityReport;
    sharpness: number;
    brightness: number;
}

export interface EngineInfo {
    landmarkerDelegate: string;
    embeddingBackend: string;
    modelId: string;
}

/**
 * Everything runs on the device: MediaPipe landmarks (GPU) -> pose + 5 points -> aligned 112x112 crop
 * (canvas warp) -> quality gate -> ArcFace embedding (ONNX Runtime Web, worker). Only the 512-d
 * vector ever leaves the browser.
 */
export class FaceEngine {
    private readonly cropCtx = createCropContext(FACE_CONFIG.embedding.inputSize);
    private lastVideoTs = -1;
    private imageLandmarker: Promise<LandmarkerHandle> | null = null;

    private constructor(private readonly videoLandmarker: LandmarkerHandle, private readonly embedder: EmbedderClient) { }

    static async create(): Promise<FaceEngine> {
        const [landmarker, embedder] = await Promise.all([
            createLandmarker({ ...FACE_CONFIG.landmarker, runningMode: "VIDEO" }),
            EmbedderClient.create({
                modelUrl: FACE_CONFIG.embedding.modelUrl,
                wasmBase: FACE_CONFIG.ortWasmBase,
                inputSize: FACE_CONFIG.embedding.inputSize,
            }),
        ]);
        return new FaceEngine(landmarker, embedder);
    }

    get info(): EngineInfo {
        return {
            landmarkerDelegate: this.videoLandmarker.delegate,
            embeddingBackend: this.embedder.backend,
            modelId: FACE_CONFIG.embedding.modelId,
        };
    }

    get embedderBusy(): boolean {
        return this.embedder.busy;
    }

    /** Landmarks + pose for the current video frame (synchronous, a few ms on GPU). */
    detect(video: HTMLVideoElement, timestampMs: number): FaceGeometry | null {
        // MediaPipe requires strictly increasing timestamps
        const ts = Math.max(timestampMs, this.lastVideoTs + 1);
        this.lastVideoTs = ts;
        return detectInVideo(this.videoLandmarker, video, ts);
    }

    async detectImage(image: HTMLImageElement | HTMLCanvasElement | ImageBitmap, width: number, height: number): Promise<FaceGeometry | null> {
        this.imageLandmarker ??= createLandmarker({ ...FACE_CONFIG.landmarker, runningMode: "IMAGE" });
        return detectInImage(await this.imageLandmarker, image, width, height);
    }

    align(source: CanvasImageSource, geometry: FaceGeometry, frameWidth: number, frameHeight: number,
        thresholds: QualityThresholds = DEFAULT_QUALITY): AlignedFace {
        const size = FACE_CONFIG.embedding.inputSize;
        const crop = warpFace(this.cropCtx, source, alignmentMatrix(geometry.points5, size), size);
        const gray = toGray(crop.data, size, size);
        const sharpness = laplacianVariance(gray, size, size);
        const brightness = meanValue(gray);
        const quality = assessQuality({ box: geometry.box, frameWidth, frameHeight, interOcular: geometry.interOcular, sharpness, brightness }, thresholds);
        return { geometry, crop, quality, sharpness, brightness };
    }

    /** Note: transfers `face.crop`'s buffer, the ImageData is unusable afterwards. */
    embed(face: AlignedFace): Promise<EmbeddingResult> {
        return this.embedder.embed(face.crop, FACE_CONFIG.embedding.flipTTA);
    }

    dispose(): void {
        this.embedder.dispose();
        this.videoLandmarker.landmarker.close();
        void this.imageLandmarker?.then(h => h.landmarker.close());
    }
}
