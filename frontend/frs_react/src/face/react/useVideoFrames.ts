"use client";

import { RefObject, useEffect, useEffectEvent } from "react";
import { onVideoFrame } from "../runtime/camera";

/**
 * Runs `onFrame` for every new camera frame while `active`. The callback is an effect event,
 * so it always sees current props/state (no stale closures) without restarting the loop.
 */
export function useVideoFrames(videoRef: RefObject<HTMLVideoElement | null>, active: boolean,
    onFrame: (video: HTMLVideoElement, timestampMs: number) => void | Promise<void>) {
    const handleFrame = useEffectEvent((video: HTMLVideoElement, ts: number) => onFrame(video, ts));

    useEffect(() => {
        const video = videoRef.current;
        if (!active || !video) return;
        return onVideoFrame(video, ts => handleFrame(video, ts));
    }, [videoRef, active]);
}
