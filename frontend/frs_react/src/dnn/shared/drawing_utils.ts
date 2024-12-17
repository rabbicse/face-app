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


    // Top-left corner
    ctx.beginPath();
    ctx.moveTo(box.xMin, box.yMin);
    ctx.lineTo(box.xMin + 15, box.yMin);
    ctx.moveTo(box.xMin, box.yMin);
    ctx.lineTo(box.xMin, box.yMin + 15);
    ctx.stroke();

    // Bottom-right corner
    ctx.beginPath();
    ctx.moveTo(box.xMax, box.yMax);
    ctx.lineTo(box.xMax - 15, box.yMax);
    ctx.moveTo(box.xMax, box.yMax);
    ctx.lineTo(box.xMax, box.yMax - 15);
    ctx.stroke();

    // Top-right corner
    ctx.beginPath();
    ctx.moveTo(box.xMax - 15, box.yMin);
    ctx.lineTo(box.xMax, box.yMin);
    ctx.moveTo(box.xMax, box.yMin);
    ctx.lineTo(box.xMax, box.yMin + 15);
    ctx.stroke();

    // Bottom-left corner
    ctx.beginPath();
    ctx.moveTo(box.xMin, box.yMax - 15);
    ctx.lineTo(box.xMin, box.yMax);
    ctx.moveTo(box.xMin, box.yMax);
    ctx.lineTo(box.xMin + 15, box.yMax);
    ctx.stroke();

    // Draw landmarks
    ctx.fillStyle = 'blue';
    for (let j = 0; j < landmarks.length; j++) {
        const x = Math.abs(landmarks[j].x);
        const y = Math.abs(landmarks[j].y);
        ctx.fillRect(x, y, 5, 5);
    }
}