/** Webcam helpers: sensible constraints, clean shutdown, and a per-new-frame loop. */

export interface CameraOptions {
    width?: number;
    height?: number;
    frameRate?: number;
    facingMode?: "user" | "environment";
    deviceId?: string;
}

/**
 * 640x480 @ 30 fps is the sweet spot: faces at arm's length have 80-150 px inter-ocular
 * distance (plenty for a 112 px crop) while keeping per-frame work small.
 */
export async function startCamera(video: HTMLVideoElement, opts: CameraOptions = {}): Promise<MediaStream> {
    if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera API unavailable. Use https:// or http://localhost.");
    }
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
            width: { ideal: opts.width ?? 640 },
            height: { ideal: opts.height ?? 480 },
            frameRate: { ideal: opts.frameRate ?? 30, max: 30 },
            ...(opts.deviceId ? { deviceId: { exact: opts.deviceId } } : { facingMode: opts.facingMode ?? "user" }),
        },
    });
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    await new Promise<void>((resolve, reject) => {
        if (video.readyState >= 2) return resolve();
        video.onloadeddata = () => resolve();
        video.onerror = () => reject(new Error("video element error"));
    });
    await video.play();
    return stream;
}

export function stopCamera(video: HTMLVideoElement | null, stream: MediaStream | null): void {
    stream?.getTracks().forEach(t => t.stop());
    if (video) {
        video.pause();
        video.srcObject = null;
    }
}

export function describeCameraError(e: unknown): string {
    const name = (e as { name?: string })?.name;
    if (name === "NotAllowedError") return "Camera permission was denied. Allow camera access in the browser and reload.";
    if (name === "NotFoundError" || name === "OverconstrainedError") return "No suitable camera was found.";
    if (name === "NotReadableError") return "The camera is in use by another application.";
    return e instanceof Error ? e.message : String(e);
}

type VideoWithRVFC = HTMLVideoElement & {
    requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
    cancelVideoFrameCallback?: (h: number) => void;
};

/**
 * Calls `onFrame` once per NEW decoded video frame (requestVideoFrameCallback) instead of
 * per display refresh (requestAnimationFrame): on a 120 Hz screen with a 30 fps camera
 * the old rAF loop processed every frame 4 times. Falls back to rAF where unsupported.
 * `onFrame` may be async; the next frame is not processed until it resolves.
 */
export function frameLoop(video: HTMLVideoElement, onFrame: (ts: number) => void | Promise<void>): () => void {
    const v = video as VideoWithRVFC;
    let stopped = false;
    let handle = 0;
    let lastMediaTime = -1;
    const useRVFC = typeof v.requestVideoFrameCallback === "function";

    const schedule = () => {
        if (stopped) return;
        if (useRVFC) handle = v.requestVideoFrameCallback!((now) => tick(now));
        else handle = requestAnimationFrame((now) => {
            if (video.currentTime === lastMediaTime) return schedule(); // same frame as before
            lastMediaTime = video.currentTime;
            tick(now);
        });
    };
    const tick = async (now: number) => {
        if (stopped) return;
        try { await onFrame(now); } catch (e) { console.error("[face] frame error", e); }
        schedule();
    };
    schedule();
    return () => {
        stopped = true;
        if (useRVFC) v.cancelVideoFrameCallback?.(handle); else cancelAnimationFrame(handle);
    };
}
