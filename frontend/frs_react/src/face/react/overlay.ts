import { FaceGeometry } from "../core/landmarks";

export type OverlayTone = "neutral" | "good" | "warn" | "success";

const COLORS: Record<OverlayTone, string> = {
    neutral: "rgba(255,255,255,0.8)",
    good: "rgba(34,197,94,0.95)",
    warn: "rgba(250,204,21,0.95)",
    success: "rgba(59,130,246,0.95)",
};

/** Draws the face box corners, alignment points and an optional progress ring (0..1). */
export function drawOverlay(canvas: HTMLCanvasElement, width: number, height: number,
    face: FaceGeometry | null, tone: OverlayTone, progress?: number) {
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);

    const color = COLORS[tone];
    const cx = width / 2, cy = height / 2;
    const rx = Math.min(width, height) * 0.3, ry = rx * 1.3;

    // guide ellipse with progress
    ctx.lineWidth = 4;
    ctx.strokeStyle = "rgba(255,255,255,0.25)";
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();
    if (progress !== undefined && progress > 0) {
        ctx.strokeStyle = COLORS.success;
        ctx.beginPath();
        ctx.ellipse(cx, cy, rx, ry, 0, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(1, progress));
        ctx.stroke();
    }

    if (!face) return;
    const { x, y, width: w, height: h } = face.box;
    const c = Math.min(w, h) * 0.15;
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (const [px, py, dx, dy] of [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]]) {
        ctx.moveTo(px + dx * c, py);
        ctx.lineTo(px, py);
        ctx.lineTo(px, py + dy * c);
    }
    ctx.stroke();

    ctx.fillStyle = color;
    for (const p of face.points5) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
        ctx.fill();
    }
}
