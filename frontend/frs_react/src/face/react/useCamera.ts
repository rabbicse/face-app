"use client";

import { RefObject, useEffect, useState } from "react";
import { FACE_CONFIG } from "../config";
import { startCamera, stopCamera } from "../runtime/camera";

export function useCamera(videoRef: RefObject<HTMLVideoElement | null>, enabled = true) {
    const [ready, setReady] = useState(false);
    const [error, setError] = useState<string | null>(null);

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
            .catch(e => setError(e instanceof Error ? e.message : String(e)));

        return () => {
            cancelled = true;
            stopCamera(video, stream);
            setReady(false);
        };
    }, [videoRef, enabled]);

    return { ready, error };
}
