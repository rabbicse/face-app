"use client";

import { useCallback, useEffect, useState } from "react";
import { useFaceEngine } from "../hooks/useFaceEngine";
import { FaceCapture } from "./FaceCapture";
import { CaptureMode } from "../runtime/controller";
import type { BenchmarkResult } from "../runtime/protocol";
import { Button } from "@/components/ui/button";

/** Live mode that embeds every frame it can (to measure sustained throughput). */
class FloodMode implements CaptureMode {
    count = 0;
    started = performance.now();
    onTrack(track: Parameters<CaptureMode["onTrack"]>[0], ctx: Parameters<CaptureMode["onTrack"]>[1]) {
        const secs = (performance.now() - this.started) / 1000;
        ctx.state.set({ message: track.primary ? `${this.count} embeddings · ${(this.count / secs).toFixed(1)}/s sustained` : "No face", tone: "info" });
        return !!track.primary;
    }
    onEmbedding(r: unknown) { if (r) this.count++; }
}

export function Benchmark() {
    const engine = useFaceEngine();
    const [res, setRes] = useState<BenchmarkResult | null>(null);
    const [running, setRunning] = useState(false);
    const createMode = useCallback(() => new FloodMode(), []);
    // read browser-only values after mount (avoids a hydration mismatch)
    const [coi, setCoi] = useState("…");
    useEffect(() => setCoi(String(window.crossOriginIsolated)), []);

    const run = async () => {
        if (engine.status !== "ready") return;
        setRunning(true);
        try { setRes(await engine.embedder.benchmark(30)); } finally { setRunning(false); }
    };

    const info = engine.status === "ready" ? engine.embedder.info : null;
    return (
        <div className="w-full max-w-2xl space-y-4">
            <h1 className="text-2xl font-semibold tracking-tight">On-device benchmark</h1>
            <div className="rounded-xl border p-4 font-mono text-sm">
                <div>ONNX backend: {info?.backend ?? "…"} · WASM threads: {info?.threads ?? "…"} · model load {info ? info.loadMs.toFixed(0) : "…"} ms</div>
                <div>crossOriginIsolated: {coi} · tracker delegate: {engine.status === "ready" ? engine.tracker.delegate : "…"}</div>
                {res && <div className="mt-2">SCRFD 192²: {res.detectMs.toFixed(2)} ms · ArcFace 112²: {res.embedMs.toFixed(2)} ms · total ≈ {(res.detectMs + res.embedMs).toFixed(1)} ms/face</div>}
                <Button className="mt-3" size="sm" onClick={run} disabled={engine.status !== "ready" || running}>{running ? "Running…" : "Run synthetic benchmark (30×)"}</Button>
            </div>
            <FaceCapture createMode={createMode} showStats drawLandmarks />
        </div>
    );
}
