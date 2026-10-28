"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, Loader2, RotateCcw } from "lucide-react";
import { useFaceEngine } from "../hooks/useFaceEngine";
import { CaptureMode, CaptureState, FaceCaptureController } from "../runtime/controller";
import { describeCameraError, startCamera, stopCamera } from "../runtime/camera";
import { cn } from "@/lib/utils";

const EMPTY: CaptureState = {
    status: "idle", message: "", arrow: null, tone: "info", progress: 0, faceVisible: false,
    stats: { fps: 0, trackMs: 0, detectMs: 0, embedMs: 0, roundTripMs: 0, backend: "", delegate: "" },
};
const noopSubscribe = () => () => { };

export interface FaceCaptureProps {
    /** called once per mount; must return a fresh mode */
    createMode: () => CaptureMode;
    onController?: (c: FaceCaptureController | null) => void;
    showStats?: boolean;
    drawLandmarks?: boolean;
    className?: string;
}

/**
 * Camera view + guide ring. All per-frame work happens in FaceCaptureController;
 * this component only re-renders when the visible guidance actually changes.
 */
export function FaceCapture({ createMode, onController, showStats = false, drawLandmarks = false, className }: FaceCaptureProps) {
    const engine = useFaceEngine();
    const videoRef = useRef<HTMLVideoElement>(null);
    const overlayRef = useRef<HTMLCanvasElement>(null);
    const [controller, setController] = useState<FaceCaptureController | null>(null);
    const [cameraError, setCameraError] = useState<string | null>(null);
    const createModeRef = useRef(createMode);
    const onControllerRef = useRef(onController);
    useEffect(() => { onControllerRef.current = onController; });

    useEffect(() => {
        if (engine.status !== "ready") return;
        const video = videoRef.current!;
        let stream: MediaStream | null = null;
        let ctrl: FaceCaptureController | null = null;
        let disposed = false;
        startCamera(video)
            .then(s => {
                stream = s;
                if (disposed) return stopCamera(video, s);
                ctrl = new FaceCaptureController(video, overlayRef.current, engine.tracker, engine.embedder, createModeRef.current(), { drawLandmarks });
                ctrl.start();
                setController(ctrl);
                onControllerRef.current?.(ctrl);
            })
            .catch(e => setCameraError(describeCameraError(e)));
        return () => {
            // the old components returned stopVideo() (calling it immediately) instead of
            // a cleanup function, so the camera kept running after leaving the page
            disposed = true;
            ctrl?.stop();
            stopCamera(video, stream);
            onControllerRef.current?.(null);
            setController(null);
        };
    }, [engine, drawLandmarks]);

    const state = useSyncExternalStore(
        controller ? controller.state.subscribe : noopSubscribe,
        controller ? controller.state.get : () => EMPTY,
        () => EMPTY,
    );

    const ringColor = state.tone === "good" ? "#22c55e" : state.tone === "warn" ? "#f59e0b" : state.tone === "error" ? "#ef4444" : "#e4e4e7";
    const R = 46, C = 2 * Math.PI * R;
    const Arrow = { left: ArrowLeft, right: ArrowRight, up: ArrowUp, down: ArrowDown, straighten: RotateCcw }[state.arrow ?? "left"];

    return (
        <div className={cn("relative mx-auto aspect-[4/3] w-full max-w-2xl overflow-hidden rounded-2xl bg-zinc-950", className)}>
            {/* mirrored preview; processing always uses the un-mirrored frame */}
            <video ref={videoRef} className="absolute inset-0 h-full w-full -scale-x-100 object-cover" playsInline muted />
            <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100 object-cover" />

            {/* guide ring with progress */}
            <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet" className="pointer-events-none absolute inset-0 h-full w-full">
                <defs>
                    <mask id="face-hole">
                        <rect width="100" height="100" fill="white" />
                        <ellipse cx="50" cy="50" rx="30" ry="39" fill="black" />
                    </mask>
                </defs>
                <rect width="100" height="100" fill="rgba(0,0,0,0.45)" mask="url(#face-hole)" />
                <ellipse cx="50" cy="50" rx="30" ry="39" fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth="0.8" />
                <circle cx="50" cy="50" r={R} fill="none" stroke={ringColor} strokeWidth="1.6" strokeLinecap="round"
                    strokeDasharray={`${C * state.progress} ${C}`} transform="rotate(-90 50 50)"
                    style={{ transition: "stroke-dasharray 300ms ease, stroke 200ms" }} />
            </svg>

            {state.arrow && state.status === "running" && (
                <div className={cn("absolute flex h-14 w-14 items-center justify-center rounded-full bg-white/90 text-zinc-900 shadow-lg animate-pulse",
                    state.arrow === "left" && "left-4 top-1/2 -translate-y-1/2",
                    state.arrow === "right" && "right-4 top-1/2 -translate-y-1/2",
                    state.arrow === "up" && "left-1/2 top-4 -translate-x-1/2",
                    state.arrow === "down" && "bottom-20 left-1/2 -translate-x-1/2",
                    state.arrow === "straighten" && "left-1/2 top-4 -translate-x-1/2")}>
                    <Arrow className="h-7 w-7" />
                </div>
            )}

            <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-4 text-center">
                {engine.status === "loading" && (
                    <div className="space-y-2 text-sm text-zinc-200">
                        <div className="flex items-center justify-center gap-2"><Loader2 className="h-4 w-4 animate-spin" />{engine.detail}</div>
                        <div className="mx-auto h-1 w-56 overflow-hidden rounded bg-white/20">
                            <div className="h-full bg-white transition-all" style={{ width: `${Math.round(engine.progress * 100)}%` }} />
                        </div>
                    </div>
                )}
                {engine.status === "error" && <p className="text-sm text-red-300">Could not start face engine: {engine.error}</p>}
                {cameraError && <p className="text-sm text-red-300">{cameraError}</p>}
                {engine.status === "ready" && !cameraError && (
                    <p className={cn("text-base font-medium", state.tone === "error" ? "text-red-300" : state.tone === "warn" ? "text-amber-200" : "text-white")}>
                        {state.message || "Starting camera…"}
                    </p>
                )}
            </div>

            {showStats && engine.status === "ready" && (
                <div className="absolute left-2 top-2 rounded-md bg-black/60 px-2 py-1 font-mono text-[11px] leading-4 text-zinc-200">
                    <div>{state.stats.fps.toFixed(0)} fps · tracker {state.stats.trackMs.toFixed(1)} ms ({state.stats.delegate})</div>
                    <div>kps {state.stats.detectMs.toFixed(1)} ms · embed {state.stats.embedMs.toFixed(1)} ms ({state.stats.backend})</div>
                </div>
            )}
        </div>
    );
}
