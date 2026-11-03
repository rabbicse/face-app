import { FaceGeometry } from "../core/landmarks";
import type { HeadPose } from "../core/pose";

export type OverlayTone = "neutral" | "good" | "warn" | "success";

// Drawn over live video, so these are bright versions of the app's brand and state colours.
const COLORS: Record<OverlayTone, string> = {
    neutral: "rgba(255,255,255,0.85)",
    good: "hsla(155, 75%, 52%, 0.95)",
    warn: "hsla(40, 96%, 58%, 0.95)",
    success: "hsla(186, 85%, 58%, 0.98)",
};
const PANEL = "hsla(200, 45%, 10%, 0.55)";
const PANEL_LINE = "rgba(255,255,255,0.18)";
const AXIS_COLORS = { x: "hsla(4, 90%, 66%, 1)", y: "hsla(140, 70%, 58%, 1)", z: "hsla(186, 85%, 62%, 1)" };

export interface PoseAim {
    /** current head pose relative to the user's own frontal pose */
    current: HeadPose;
    /** the yaw and pitch window the current step wants */
    target: { yaw: [number, number]; pitch: [number, number] };
}

export interface OverlayOptions {
    /** 0..1 progress ring around the guide ellipse */
    progress?: number;
    /** the 5 alignment points (diagnostic pages only) */
    points?: boolean;
    /** small 3D axes showing the head's orientation */
    axes?: boolean;
    /** pad that shows where to point the head (registration) */
    aim?: PoseAim;
}

/**
 * Face box corners, guide ellipse with optional progress, and optional pose widgets.
 * The canvas is mirrored with CSS like the preview, so everything is drawn in non-mirrored camera
 * coordinates: a widget drawn on the canvas' left shows on the screen's right, and the face's +yaw
 * (turned to the image's right) shows as the user's own left, matching what they see.
 */
export function drawOverlay(canvas: HTMLCanvasElement, width: number, height: number,
    face: FaceGeometry | null, tone: OverlayTone, options: OverlayOptions = {}) {
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    const color = COLORS[tone];
    const cx = width / 2, cy = height / 2;
    const rx = Math.min(width, height) * 0.3, ry = rx * 1.3;

    // guide ellipse with progress
    ctx.lineWidth = 4;
    ctx.strokeStyle = "rgba(255,255,255,0.3)";
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();
    const { progress } = options;
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

    if (options.points) {
        ctx.fillStyle = color;
        for (const p of face.points5) {
            ctx.beginPath();
            ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    const unit = Math.min(width, height);
    const margin = unit * 0.04;
    if (options.axes) {
        const r = unit * 0.085;
        drawAxes(ctx, margin + r, margin + r, r, face.pose); // screen top-right
    }
    if (options.aim) {
        const size = unit * 0.24;
        drawAim(ctx, width - margin - size, height - margin - size, size, options.aim, color); // screen bottom-left
    }
}

/** Unit vectors of the head's own axes in camera space (x right, y up, z towards the camera). */
export function headAxes(pose: HeadPose): { x: number[]; y: number[]; z: number[] } {
    const rad = Math.PI / 180;
    const [cy, sy] = [Math.cos(pose.yaw * rad), Math.sin(pose.yaw * rad)];
    const [cp, sp] = [Math.cos(pose.pitch * rad), Math.sin(pose.pitch * rad)];
    const [cr, sr] = [Math.cos(pose.roll * rad), Math.sin(pose.roll * rad)];
    // R = Rz(roll) * Ry(yaw) * Rx(pitch); yaw > 0 turns z towards +x, pitch > 0 tilts z up, roll > 0 is CCW
    const Rx = [[1, 0, 0], [0, cp, sp], [0, -sp, cp]];
    const Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]];
    const Rz = [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]];
    const mul = (a: number[][], b: number[][]) => a.map(row => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)));
    const R = mul(Rz, mul(Ry, Rx));
    const col = (j: number) => [R[0][j], R[1][j], R[2][j]];
    return { x: col(0), y: col(1), z: col(2) };
}

function drawAxes(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number, pose: HeadPose) {
    ctx.fillStyle = PANEL;
    ctx.strokeStyle = PANEL_LINE;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    const axes = headAxes(pose);
    const len = r * 0.72;
    // back to front, so the axis pointing at the camera is drawn on top
    const order = (["x", "y", "z"] as const).slice().sort((a, b) => axes[a][2] - axes[b][2]);
    for (const k of order) {
        const [vx, vy, vz] = axes[k];
        const ex = cx + vx * len, ey = cy - vy * len;
        ctx.globalAlpha = vz < -0.2 ? 0.45 : 1; // axes pointing away read as behind
        ctx.strokeStyle = AXIS_COLORS[k];
        ctx.fillStyle = AXIS_COLORS[k];
        ctx.lineWidth = k === "z" ? 3.5 : 2.5;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(ex, ey, k === "z" ? 4 : 3, 0, Math.PI * 2);
        ctx.fill();
    }
    ctx.globalAlpha = 1;
}

/** Square pad: the target window as a zone, the head's direction as a dot to steer into it. */
function drawAim(ctx: CanvasRenderingContext2D, left: number, top: number, size: number, aim: PoseAim, color: string) {
    const RANGE = 45; // degrees from the centre to the pad's edge
    const half = size / 2, cx = left + half, cy = top + half;
    const toX = (yaw: number) => cx + (Math.max(-RANGE, Math.min(RANGE, yaw)) / RANGE) * half * 0.9;
    const toY = (pitch: number) => cy - (Math.max(-RANGE, Math.min(RANGE, pitch)) / RANGE) * half * 0.9;

    ctx.fillStyle = PANEL;
    ctx.strokeStyle = PANEL_LINE;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(left, top, size, size, size * 0.14);
    ctx.fill();
    ctx.stroke();
    // crosshair
    ctx.beginPath();
    ctx.moveTo(cx, top + size * 0.12); ctx.lineTo(cx, top + size * 0.88);
    ctx.moveTo(left + size * 0.12, cy); ctx.lineTo(left + size * 0.88, cy);
    ctx.stroke();

    // target window
    const { yaw, pitch } = aim.target;
    const zx0 = Math.min(toX(yaw[0]), toX(yaw[1])), zx1 = Math.max(toX(yaw[0]), toX(yaw[1]));
    const zy0 = Math.min(toY(pitch[0]), toY(pitch[1])), zy1 = Math.max(toY(pitch[0]), toY(pitch[1]));
    ctx.fillStyle = "hsla(186, 85%, 58%, 0.22)";
    ctx.strokeStyle = COLORS.success;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(zx0, zy0, zx1 - zx0, zy1 - zy0, 6);
    ctx.fill();
    ctx.stroke();

    // current direction, with a guide line to the window's centre while outside it
    const px = toX(aim.current.yaw), py = toY(aim.current.pitch);
    const inside = px >= zx0 && px <= zx1 && py >= zy0 && py <= zy1;
    if (!inside) {
        ctx.strokeStyle = "rgba(255,255,255,0.55)";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo((zx0 + zx1) / 2, (zy0 + zy1) / 2);
        ctx.stroke();
        ctx.setLineDash([]);
    }
    ctx.fillStyle = color;
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px, py, Math.max(5, size * 0.055), 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
}
