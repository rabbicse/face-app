export interface CameraOptions {
    width: number;
    height: number;
    frameRate: number;
    facingMode?: "user" | "environment";
}

export async function startCamera(video: HTMLVideoElement, opts: CameraOptions): Promise<MediaStream> {
    if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access is not available (a secure https:// or localhost origin is required)");
    }
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
            facingMode: opts.facingMode ?? "user",
            width: { ideal: opts.width },
            height: { ideal: opts.height },
            frameRate: { ideal: opts.frameRate, max: opts.frameRate },
        },
    });
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    if (video.readyState < HTMLMediaElement.HAVE_METADATA) {
        await new Promise<void>(resolve => video.addEventListener("loadedmetadata", () => resolve(), { once: true }));
    }
    await video.play();
    return stream;
}

export function stopCamera(video: HTMLVideoElement | null, stream: MediaStream | null): void {
    stream?.getTracks().forEach(t => t.stop());
    if (video) {
        video.srcObject = null;
    }
}

type VideoWithFrameCallback = HTMLVideoElement & {
    requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
    cancelVideoFrameCallback?: (handle: number) => void;
};

/**
 * Calls `onFrame` once per *new* decoded video frame (requestVideoFrameCallback), falling back to
 * requestAnimationFrame. Frames that arrive while `onFrame` is still running are skipped, so slow
 * processing never queues up work. Returns a stop function.
 */
export function onVideoFrame(video: HTMLVideoElement, onFrame: (timestampMs: number) => void | Promise<void>): () => void {
    const v = video as VideoWithFrameCallback;
    let stopped = false;
    let busy = false;
    let handle = 0;
    const useRvfc = typeof v.requestVideoFrameCallback === "function";

    const schedule = () => {
        if (stopped) return;
        handle = useRvfc ? v.requestVideoFrameCallback!(tick) : requestAnimationFrame(tick);
    };

    const tick = (now: number) => {
        if (stopped) return;
        if (!busy && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
            busy = true;
            Promise.resolve(onFrame(now))
                .catch(e => console.error("frame processing failed", e))
                .finally(() => { busy = false; });
        }
        schedule();
    };

    schedule();
    return () => {
        stopped = true;
        if (useRvfc) v.cancelVideoFrameCallback?.(handle);
        else cancelAnimationFrame(handle);
    };
}
