import { Face } from "./types";

// Draw detections
export function drawDetection(ctx: CanvasRenderingContext2D, detection: Face) {
    const box = detection.box;
    const landmarks = detection.keypoints;


    // draw rectangle background
    ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
    ctx.fillRect(
        Math.abs(box.xMin),
        Math.abs(box.yMin),
        Math.abs(box.width),
        Math.abs(box.height)
    );

    // Draw rectangle outline
    ctx.strokeStyle = 'rgba(100, 149, 237, 0.8)'; // Set stroke color and transparency
    ctx.lineWidth = 2; // Set the line width for better visibility
    ctx.strokeRect(
        Math.abs(box.xMin),
        Math.abs(box.yMin),
        Math.abs(box.width),
        Math.abs(box.height)
    );

    // Draw landmarks
    ctx.fillStyle = 'blue';
    for (let j = 0; j < landmarks.length; j++) {
        const x = Math.abs(landmarks[j].x);
        const y = Math.abs(landmarks[j].y);
        ctx.fillRect(x, y, 5, 5);
    }
}