"use client";

import { RefObject, useEffect, useRef } from "react";
import { onVideoFrame } from "../runtime/camera";

/**
 * Runs `onFrame` for every new camera frame while `active`. The latest callback is kept in a ref,
 * so it always sees current props/state (no stale closures) without restarting the loop.
 */
export function useVideoFrames(videoRef: RefObject<HTMLVideoElement | null>, active: boolean,
    onFrame: (video: HTMLVideoElement, timestampMs: number) => void | Promise<void>) {
    const callback = useRef(onFrame);
    callback.current = onFrame;

    useEffect(() => {
        const video = videoRef.current;
        if (!active || !video) return;
        return onVideoFrame(video, ts => callback.current(video, ts));
    }, [videoRef, active]);
}
