// Function to draw landmarks
function drawLandmark(ctx, landmark, color, width, height) {
    const x = landmark.x * width;
    const y = landmark.y * height;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
}

// Draw detections
function drawDetections(ctx, detections, imgWidth, imgHeight) {
    const rectColor = 'rgba(224, 128, 20, 0.8)';
    const rectLineWidth = 2;

    detections.forEach(detection => {
        const bbox = detection.bbox;
        const landmarks = detection.landmarks;

        const xMin = bbox.x_min * imgWidth;
        const yMin = bbox.y_min * imgHeight;
        const xMax = bbox.x_max * imgWidth;
        const yMax = bbox.y_max * imgHeight;

        // Draw bounding box
        ctx.strokeStyle = rectColor;
        ctx.lineWidth = rectLineWidth;
        ctx.strokeRect(xMin, yMin, xMax - xMin, yMax - yMin);

        // Draw score
        ctx.fillStyle = 'white';
        ctx.font = '16px Arial';
        ctx.fillText(`Score: ${bbox.score.toFixed(2)}`, xMin + 10, yMin + 20);

        // Draw landmarks
        // drawLandmark(ctx, landmarks.left_eye, 'red', imgWidth, imgHeight);
        // drawLandmark(ctx, landmarks.right_eye, 'red', imgWidth, imgHeight);
        // drawLandmark(ctx, landmarks.nose, 'green', imgWidth, imgHeight);
        // drawLandmark(ctx, landmarks.left_lip, 'blue', imgWidth, imgHeight);
        // drawLandmark(ctx, landmarks.right_lip, 'blue', imgWidth, imgHeight);
    });
}