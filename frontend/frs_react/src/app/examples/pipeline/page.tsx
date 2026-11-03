"use client"

import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { dot } from "@/face/core/embedding";
import { classifyPose } from "@/face/core/pose";
import { BlinkDetector } from "@/face/core/liveness";
import { FaceCamera } from "@/face/react/FaceCamera";
import { drawOverlay } from "@/face/react/overlay";
import { useCamera } from "@/face/react/useCamera";
import { useFaceEngine } from "@/face/react/useFaceEngine";
import { useVideoFrames } from "@/face/react/useVideoFrames";

interface Stats {
    fps: number;
    detectMs: number;
    alignMs: number;
    embedMs: number;
    yaw: number;
    pitch: number;
    roll: number;
    poseLabel: string;
    interOcular: number;
    sharpness: number;
    brightness: number;
    issues: string;
    eyesClosed: number;
    blinks: number;
    similarity: number | null;
}

const fmt = (v: number, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : "-");

export default function PipelineInspectorPage() {
    const videoRef = useRef<HTMLVideoElement>(null);
    const overlayRef = useRef<HTMLCanvasElement>(null);
    const cropRef = useRef<HTMLCanvasElement>(null);
    const { engine, error } = useFaceEngine();
    const { ready } = useCamera(videoRef);

    const [stats, setStats] = useState<Stats | null>(null);
    const latest = useRef<Float32Array | null>(null);
    const reference = useRef<Float32Array | null>(null);
    const frameTimes = useRef<number[]>([]);
    const embedMs = useRef(NaN);
    const lastUi = useRef(0);
    const blink = useRef(new BlinkDetector());

    useVideoFrames(videoRef, !!engine && ready, (video, ts) => {
        if (!engine) return;
        const t0 = performance.now();
        const face = engine.detect(video, ts);
        const detectMs = performance.now() - t0;
        frameTimes.current = [...frameTimes.current.filter(t => ts - t < 1000), ts];

        if (overlayRef.current) drawOverlay(overlayRef.current, video.videoWidth, video.videoHeight, face, face ? "good" : "neutral", { points: true, axes: true });
        if (!face) return;
        if (face.eyesClosed !== undefined) blink.current.update(face.eyesClosed, ts);

        const t1 = performance.now();
        const aligned = engine.align(video, face, video.videoWidth, video.videoHeight);
        const alignMs = performance.now() - t1;
        cropRef.current?.getContext("2d")?.putImageData(aligned.crop, 0, 0);
        if (!engine.embedderBusy) {
            void engine.embed(aligned).then(r => {
                latest.current = r.embedding;
                embedMs.current = r.ms;
            });
        }

        // refresh the panel at ~8 Hz, not on every frame
        if (ts - lastUi.current < 125) return;
        lastUi.current = ts;
        setStats({
            fps: frameTimes.current.length,
            detectMs,
            alignMs,
            embedMs: embedMs.current,
            ...face.pose,
            poseLabel: classifyPose(face.pose) ?? "-",
            interOcular: face.interOcular,
            sharpness: aligned.sharpness,
            brightness: aligned.brightness,
            issues: aligned.quality.issues.join(", ") || "none",
            eyesClosed: face.eyesClosed ?? NaN,
            blinks: blink.current.count,
            similarity: reference.current && latest.current ? dot(reference.current, latest.current) : null,
        });
    });

    const rows: [string, string][] = stats ? [
        ["Landmarker / embedding backend", engine ? `${engine.info.landmarkerDelegate} / ${engine.info.embeddingBackend}` : "-"],
        ["Model init: landmarker / embedder (ms)", engine ? `${fmt(engine.info.landmarkerInitMs, 0)} / ${fmt(engine.info.embedderInitMs, 0)}` : "-"],
        ["Camera FPS processed", String(stats.fps)],
        ["Landmarks (ms)", fmt(stats.detectMs)],
        ["Align + quality (ms)", fmt(stats.alignMs, 2)],
        ["Embedding incl. flip TTA (ms)", fmt(stats.embedMs)],
        ["Yaw / pitch / roll (deg)", `${fmt(stats.yaw)} / ${fmt(stats.pitch)} / ${fmt(stats.roll)}`],
        ["Pose class", stats.poseLabel],
        ["Eye distance (px)", fmt(stats.interOcular, 0)],
        ["Sharpness / brightness", `${fmt(stats.sharpness, 0)} / ${fmt(stats.brightness, 0)}`],
        ["Quality issues", stats.issues],
        ["Eyes closed score / blinks", `${fmt(stats.eyesClosed, 2)} / ${stats.blinks}`],
        ["Cosine similarity to reference", stats.similarity === null ? "press 'Set reference'" : fmt(stats.similarity, 3)],
    ] : [];

    return (
        <main className="mx-auto grid w-full max-w-5xl flex-1 content-start gap-5 px-4 pb-16 pt-4 sm:px-6 md:grid-cols-[3fr_2fr]">
            <div className="space-y-1 md:col-span-2">
                <h1 className="text-3xl font-semibold tracking-tight">Live pipeline inspector</h1>
                <p className="text-muted-foreground">Each stage of the on-device pipeline, timed on this device.</p>
            </div>
            <FaceCamera ref={videoRef} overlayRef={overlayRef} className="aspect-[4/3] w-full" />
            <Card>
                <CardContent className="space-y-3 p-4 text-sm">
                    {error && <p role="alert" className="rounded-lg bg-destructive-soft px-3 py-2 font-medium text-destructive">{error}</p>}
                    {!engine && !error && <p className="text-muted-foreground">Loading models...</p>}
                    <div className="flex items-center gap-3">
                        <canvas ref={cropRef} width={112} height={112} className="rounded-lg border bg-muted" title="aligned 112x112 crop fed to the model" />
                        <Button onClick={() => { reference.current = latest.current; }} disabled={!stats}>Set reference</Button>
                    </div>
                    <table className="w-full">
                        <tbody>
                            {rows.map(([k, v]) => (
                                <tr key={k} className="border-b last:border-0">
                                    <td className="py-1.5 pr-2 text-muted-foreground">{k}</td>
                                    <td className="whitespace-nowrap py-1.5 text-right font-mono tabular-nums">{v}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </CardContent>
            </Card>
        </main>
    );
}
