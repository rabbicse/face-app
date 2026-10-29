"use client";

import { useEffect, useState } from "react";
import type { FaceEngine } from "../runtime/engine";

let enginePromise: Promise<FaceEngine> | null = null;

/**
 * Loads the models once per browser session; every page re-uses the same engine. Call early
 * (e.g. while the user types credentials) so the camera page starts instantly.
 */
export function preloadFaceEngine(): Promise<FaceEngine> {
    if (!enginePromise) {
        enginePromise = import("../runtime/engine").then(m => m.FaceEngine.create());
        enginePromise.catch(() => { enginePromise = null; }); // allow a retry
    }
    return enginePromise;
}

/** Preload when the browser is idle. */
export function usePreloadFaceEngine(): void {
    useEffect(() => {
        const w = window as Window & { requestIdleCallback?: (cb: () => void) => number };
        const start = () => { preloadFaceEngine().catch(e => console.warn("face engine preload failed", e)); };
        if (w.requestIdleCallback) w.requestIdleCallback(start);
        else setTimeout(start, 500);
    }, []);
}

export function useFaceEngine(): { engine: FaceEngine | null; error: string | null } {
    const [engine, setEngine] = useState<FaceEngine | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let alive = true;
        preloadFaceEngine()
            .then(e => { if (alive) setEngine(e); })
            .catch(e => { if (alive) setError(e instanceof Error ? e.message : String(e)); });
        return () => { alive = false; };
    }, []);

    return { engine, error };
}
