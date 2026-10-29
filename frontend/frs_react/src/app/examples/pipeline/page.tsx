"use client"

import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { dot } from "@/face/core/embedding";
import { classifyPose } from "@/face/core/pose";
import { FaceCamera } from "@/face/react/FaceCamera";
import { drawOverlay } from "@/face/react/overlay";
import { useCamera } from "@/face/react/useCamera";
import { useFaceEngine } from "@/face/react/useFaceEngine";
import { useVideoFrames } from "@/face/react/useVideoFrames";

interface Stats {
    fps: number;
    detectMs: number;
    embedMs: number;
    yaw: number;
    pitch: number;
    roll: number;
    poseLabel: string;
    interOcular: number;
    sharpness: number;
    brightness: number;
    issues: string;
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

    useVideoFrames(videoRef, !!engine && ready, (video, ts) => {
        if (!engine) return;
        const t0 = performance.now();
        const face = engine.detect(video, ts);
        const detectMs = performance.now() - t0;
        frameTimes.current = [...frameTimes.current.filter(t => ts - t < 1000), ts];

        if (overlayRef.current) drawOverlay(overlayRef.current, video.videoWidth, video.videoHeight, face, face ? "good" : "neutral");
        if (!face) return;

        const aligned = engine.align(video, face, video.videoWidth, video.videoHeight);
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
            embedMs: embedMs.current,
            ...face.pose,
            poseLabel: classifyPose(face.pose) ?? "-",
            interOcular: face.interOcular,
            sharpness: aligned.sharpness,
            brightness: aligned.brightness,
            issues: aligned.quality.issues.join(", ") || "none",
            similarity: reference.current && latest.current ? dot(reference.current, latest.current) : null,
        });
    });

    const rows: [string, string][] = stats ? [
        ["Landmarker / embedding backend", engine ? `${engine.info.landmarkerDelegate} / ${engine.info.embeddingBackend}` : "-"],
        ["Camera FPS processed", String(stats.fps)],
        ["Landmarks (ms)", fmt(stats.detectMs)],
        ["Embedding incl. flip TTA (ms)", fmt(stats.embedMs)],
        ["Yaw / pitch / roll (deg)", `${fmt(stats.yaw)} / ${fmt(stats.pitch)} / ${fmt(stats.roll)}`],
        ["Pose class", stats.poseLabel],
        ["Eye distance (px)", fmt(stats.interOcular, 0)],
        ["Sharpness / brightness", `${fmt(stats.sharpness, 0)} / ${fmt(stats.brightness, 0)}`],
        ["Quality issues", stats.issues],
        ["Cosine similarity to reference", stats.similarity === null ? "press 'Set reference'" : fmt(stats.similarity, 3)],
    ] : [];

    return (
        <main className="mx-auto grid max-w-5xl gap-4 p-6 md:grid-cols-[3fr_2fr]">
            <FaceCamera ref={videoRef} overlayRef={overlayRef} className="aspect-[4/3] w-full" />
            <Card>
                <CardContent className="space-y-3 p-4 text-sm">
                    {error && <p className="text-red-600">{error}</p>}
                    {!engine && !error && <p>Loading models...</p>}
                    <div className="flex items-center gap-3">
                        <canvas ref={cropRef} width={112} height={112} className="rounded border" title="aligned 112x112 crop fed to the model" />
                        <Button onClick={() => { reference.current = latest.current; }} disabled={!stats}>Set reference</Button>
                    </div>
                    <table className="w-full">
                        <tbody>
                            {rows.map(([k, v]) => (
                                <tr key={k} className="border-b last:border-0">
                                    <td className="py-1 pr-2 text-muted-foreground">{k}</td>
                                    <td className="py-1 font-mono">{v}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </CardContent>
            </Card>
        </main>
    );
}
