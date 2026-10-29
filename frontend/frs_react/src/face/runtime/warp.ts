import { Affine } from "../core/geometry";

type Ctx2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

export function createCropContext(size: number): Ctx2D {
    const canvas: OffscreenCanvas | HTMLCanvasElement = typeof OffscreenCanvas !== "undefined"
        ? new OffscreenCanvas(size, size)
        : Object.assign(document.createElement("canvas"), { width: size, height: size });
    const ctx = canvas.getContext("2d", { willReadFrequently: true }) as Ctx2D | null;
    if (!ctx) {
        throw new Error("2D canvas is not available");
    }
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    return ctx;
}

/**
 * Warps the face into a `size` x `size` aligned crop with one GPU accelerated drawImage call
 * (no full frame copy / readback) and returns its RGBA pixels.
 */
export function warpFace(ctx: Ctx2D, source: CanvasImageSource, m: Affine, size: number): ImageData {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, size, size);
    // canvas: x' = a*x + c*y + e, y' = b*x + d*y + f
    ctx.setTransform(m[0], m[3], m[1], m[4], m[2], m[5]);
    ctx.drawImage(source, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    return ctx.getImageData(0, 0, size, size);
}
