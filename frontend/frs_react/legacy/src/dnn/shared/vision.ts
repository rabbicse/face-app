import { BoundingBox } from "./interfaces/shapes";

export const cropFace = (image: HTMLVideoElement, box: BoundingBox): Promise<Blob> => {
    return new Promise((resolve) => {
        // Create a temporary canvas to crop the face
        const faceCanvas = document.createElement("canvas");
        const faceCtx = faceCanvas.getContext("2d");

        const xMin = Math.max(box.xMin - (box.width / 4), 0);
        const yMin = Math.max(Math.abs(box.yMin) - (box.height / 2), 0);
        const width = box.width + (2 * (box.width / 4));// + (box.width / 4) + (box.width / 4); //Math.abs(xMax - xMin);
        const height = box.height + ((box.height / 2) + (box.height / 4)); //Math.abs(yMax - yMin);

        faceCanvas.width = width;
        faceCanvas.height = height;

        // Draw the cropped face onto the temporary canvas
        faceCtx?.drawImage(
            image, // The main canvas containing the video
            xMin,
            yMin,
            width,
            height,
            0,
            0,
            faceCanvas.width,
            faceCanvas.height
        );

        return faceCanvas.toBlob((blob) => resolve(blob!), "image/jpeg");
    });
};