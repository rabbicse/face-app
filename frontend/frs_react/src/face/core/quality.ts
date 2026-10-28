/**
 * Cheap image-quality signals computed on the aligned 112x112 crop.
 * Bad crops (blurred, dark, tiny) are the #1 cause of poor recognition accuracy, so
 * they are rejected BEFORE an embedding is stored or sent anywhere.
 */
import { RgbaImage, toGray } from "./image";

export interface FaceQuality {
    /** variance of the Laplacian on the central face region; higher = sharper */
    sharpness: number;
    /** mean luma 0..255 */
    brightness: number;
    /** luma standard deviation; very low = washed out / flat */
    contrast: number;
    /** inter-ocular distance in SOURCE pixels (resolution the face was captured at) */
    interOcularPx: number;
    /** detector confidence 0..1 */
    detScore: number;
}

export interface QualityThresholds {
    minSharpness: number;
    minBrightness: number;
    maxBrightness: number;
    minContrast: number;
    minInterOcularPx: number;
    minDetScore: number;
}

/** Defaults tuned for a 640x480 webcam; registration is stricter than identification. */
// Sharpness calibrated on aligned crops (examples/python/calibrate_quality.py):
// sharp faces score 150+, ~1 px blur 15-110 (embedding cos ≈ 0.97), ~2 px blur 5-18 (cos 0.7-0.9).
export const ENROLL_QUALITY: QualityThresholds = {
    minSharpness: 40,
    minBrightness: 60,
    maxBrightness: 210,
    minContrast: 25,
    minInterOcularPx: 45,
    minDetScore: 0.6,
};

export const IDENTIFY_QUALITY: QualityThresholds = {
    minSharpness: 20,
    minBrightness: 45,
    maxBrightness: 225,
    minContrast: 18,
    minInterOcularPx: 32,
    minDetScore: 0.5,
};

/**
 * Variance of the 4-neighbour Laplacian over the central region (skips the black
 * warp border and hair/background at the edges).
 */
export function laplacianVariance(gray: Float32Array, w: number, h: number, margin = 16): number {
    let sum = 0, sum2 = 0, n = 0;
    for (let y = Math.max(1, margin); y < Math.min(h - 1, h - margin); y++) {
        for (let x = Math.max(1, margin); x < Math.min(w - 1, w - margin); x++) {
            const i = y * w + x;
            const l = gray[i - 1] + gray[i + 1] + gray[i - w] + gray[i + w] - 4 * gray[i];
            sum += l; sum2 += l * l; n++;
        }
    }
    if (n === 0) return 0;
    const mean = sum / n;
    return sum2 / n - mean * mean;
}

export function measureQuality(aligned: RgbaImage, interOcularPx: number, detScore: number): FaceQuality {
    const g = toGray(aligned);
    const { width: w, height: h } = aligned;
    // brightness/contrast on the inner face (eyes..mouth) area of the canonical crop
    let s = 0, s2 = 0, n = 0;
    for (let y = 40; y < 100; y++) {
        for (let x = 28; x < 84; x++) {
            const v = g[y * w + x];
            s += v; s2 += v * v; n++;
        }
    }
    const mean = s / n;
    return {
        sharpness: laplacianVariance(g, w, h),
        brightness: mean,
        contrast: Math.sqrt(Math.max(0, s2 / n - mean * mean)),
        interOcularPx,
        detScore,
    };
}

export type QualityIssue = "blurry" | "too_dark" | "too_bright" | "low_contrast" | "too_far" | "uncertain";

export function qualityIssues(q: FaceQuality, t: QualityThresholds): QualityIssue[] {
    const issues: QualityIssue[] = [];
    if (q.interOcularPx < t.minInterOcularPx) issues.push("too_far");
    if (q.brightness < t.minBrightness) issues.push("too_dark");
    if (q.brightness > t.maxBrightness) issues.push("too_bright");
    if (q.contrast < t.minContrast) issues.push("low_contrast");
    if (q.sharpness < t.minSharpness) issues.push("blurry");
    if (q.detScore < t.minDetScore) issues.push("uncertain");
    return issues;
}

/** Single scalar used to pick the best of several samples (higher is better). */
export function qualityScore(q: FaceQuality): number {
    const sharp = Math.min(1, q.sharpness / 300);
    const light = 1 - Math.min(1, Math.abs(q.brightness - 130) / 130);
    const size = Math.min(1, q.interOcularPx / 90);
    return 0.4 * sharp + 0.2 * light + 0.2 * size + 0.2 * q.detScore;
}

export const QUALITY_MESSAGES: Record<QualityIssue, string> = {
    blurry: "Hold still — image is blurry",
    too_dark: "Too dark — face a light source",
    too_bright: "Too bright — avoid direct light",
    low_contrast: "Low contrast — improve lighting",
    too_far: "Move closer to the camera",
    uncertain: "Make sure your whole face is visible",
};
