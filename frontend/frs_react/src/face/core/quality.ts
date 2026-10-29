import { Box } from "./landmarks";

export interface QualityMetrics {
    box: Box;
    frameWidth: number;
    frameHeight: number;
    interOcular: number;
    /** Laplacian variance of the aligned grey crop */
    sharpness: number;
    /** mean luma (0..255) of the aligned crop */
    brightness: number;
}

export interface QualityThresholds {
    /** eye distance in source pixels; the 112px template has ~35px, below that the crop is upscaled */
    minInterOcular: number;
    minSharpness: number;
    minBrightness: number;
    maxBrightness: number;
    /** fraction of the box that may lie outside the frame */
    maxOutside: number;
}

export const DEFAULT_QUALITY: QualityThresholds = {
    minInterOcular: 32,
    minSharpness: 40,
    minBrightness: 50,
    maxBrightness: 215,
    maxOutside: 0.05,
};

export type QualityIssue = "too_far" | "blurry" | "too_dark" | "too_bright" | "out_of_frame";

export interface QualityReport {
    ok: boolean;
    issues: QualityIssue[];
    /** 0..1, higher is better; used to rank samples */
    score: number;
}

export function assessQuality(m: QualityMetrics, t: QualityThresholds = DEFAULT_QUALITY): QualityReport {
    const issues: QualityIssue[] = [];

    const visibleW = Math.min(m.box.x + m.box.width, m.frameWidth) - Math.max(m.box.x, 0);
    const visibleH = Math.min(m.box.y + m.box.height, m.frameHeight) - Math.max(m.box.y, 0);
    const visible = Math.max(0, visibleW) * Math.max(0, visibleH) / Math.max(1e-6, m.box.width * m.box.height);
    if (1 - visible > t.maxOutside) issues.push("out_of_frame");
    if (m.interOcular < t.minInterOcular) issues.push("too_far");
    if (m.sharpness < t.minSharpness) issues.push("blurry");
    if (m.brightness < t.minBrightness) issues.push("too_dark");
    if (m.brightness > t.maxBrightness) issues.push("too_bright");

    const clamp = (v: number) => Math.max(0, Math.min(1, v));
    const size = clamp(m.interOcular / (2 * t.minInterOcular));
    const sharp = clamp(m.sharpness / (4 * t.minSharpness));
    const mid = (t.minBrightness + t.maxBrightness) / 2;
    const light = clamp(1 - Math.abs(m.brightness - mid) / (mid - t.minBrightness + 1e-6) * 0.5);
    const score = Math.round(clamp(visible) * (0.4 * size + 0.4 * sharp + 0.2 * light) * 1000) / 1000;

    return { ok: issues.length === 0, issues, score };
}

export const QUALITY_MESSAGES: Record<QualityIssue, string> = {
    too_far: "Move closer to the camera",
    blurry: "Hold still",
    too_dark: "Find better lighting",
    too_bright: "Too bright, avoid direct light",
    out_of_frame: "Center your face in the frame",
};
