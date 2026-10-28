"use client";
import { useEffect, useState } from "react";
import { EmbedderClient, LoadProgress } from "../runtime/embedderClient";
import { FaceTracker } from "../runtime/tracker";

export type EngineState =
    | { status: "loading"; progress: number; detail: string }
    | { status: "ready"; tracker: FaceTracker; embedder: EmbedderClient }
    | { status: "error"; error: string };

/**
 * Loads the tracker (MediaPipe) and the embedding worker (ONNX) in parallel. Both are
 * module singletons: the second page that uses them gets them instantly.
 */
export function useFaceEngine(): EngineState {
    const [state, setState] = useState<EngineState>({ status: "loading", progress: 0, detail: "Loading face models…" });

    useEffect(() => {
        let cancelled = false;
        const files: Record<string, LoadProgress> = {};
        const onProgress = (p: LoadProgress) => {
            files[p.file] = p;
            const loaded = Object.values(files).reduce((s, f) => s + f.loaded, 0);
            const total = Object.values(files).reduce((s, f) => s + f.total, 0) || 1;
            if (!cancelled) setState(s => s.status === "loading"
                ? { status: "loading", progress: loaded / total, detail: `Downloading models ${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB (first visit only)` }
                : s);
        };
        const embedder = EmbedderClient.get(onProgress);
        Promise.all([FaceTracker.load(), embedder.ready])
            .then(([tracker]) => { if (!cancelled) setState({ status: "ready", tracker, embedder }); })
            .catch(e => { if (!cancelled) setState({ status: "error", error: e instanceof Error ? e.message : String(e) }); });
        return () => {
            cancelled = true;
            EmbedderClient.removeProgressListener(onProgress);
        };
    }, []);

    return state;
}
