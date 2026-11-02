"use client";

import { RefObject, useCallback, useEffect, useState } from "react";
import { FACE_CONFIG } from "../config";
import { startCamera, stopCamera } from "../runtime/camera";
import { describeCameraError } from "./cameraErrors";

export function useCamera(videoRef: RefObject<HTMLVideoElement | null>, enabled = true) {
    const [ready, setReady] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [attempt, setAttempt] = useState(0);

    useEffect(() => {
        if (!enabled) return;
        const video = videoRef.current;
        if (!video) return;

        let stream: MediaStream | null = null;
        let cancelled = false;
        startCamera(video, FACE_CONFIG.camera)
            .then(s => {
                if (cancelled) {
                    stopCamera(null, s);
                    return;
                }
                stream = s;
                setReady(true);
            })
            .catch(e => {
                console.warn("camera failed to start", e);
                if (!cancelled) setError(describeCameraError(e));
            });

        return () => {
            cancelled = true;
            stopCamera(video, stream);
            setReady(false);
        };
    }, [videoRef, enabled, attempt]);

    const retry = useCallback(() => {
        setError(null);
        setAttempt(a => a + 1);
    }, []);

    return { ready, error, retry };
}
