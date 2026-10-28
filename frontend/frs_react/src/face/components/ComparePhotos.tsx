"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { useFaceEngine } from "../hooks/useFaceEngine";
import type { WorkerEmbedding } from "../runtime/protocol";
import { dot } from "../core/vector";
import { DEFAULT_DECISION } from "../core/identification";

function AlignedThumb({ data }: { data: ImageData }) {
    const ref = useRef<HTMLCanvasElement>(null);
    useEffect(() => { ref.current?.getContext("2d")?.putImageData(data, 0, 0); }, [data]);
    return <canvas ref={ref} width={112} height={112} className="h-28 w-28 rounded-md border" title="aligned 112x112 crop fed to ArcFace" />;
}

function Slot({ label, onFaces }: { label: string; onFaces: (f: WorkerEmbedding[] | null) => void }) {
    const engine = useFaceEngine();
    const [url, setUrl] = useState<string | null>(null);
    const [faces, setFaces] = useState<WorkerEmbedding[] | null>(null);
    const [busy, setBusy] = useState(false);
    const [err, setErr] = useState<string | null>(null);

    const pick = async (file: File | undefined) => {
        if (!file || engine.status !== "ready") return;
        setUrl(URL.createObjectURL(file));
        setBusy(true); setErr(null);
        try {
            const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
            const r = await engine.embedder.embedImage(bmp, 1);
            setFaces(r); onFaces(r);
            if (!r.length) setErr("No face found");
        } catch (e) { setErr(String(e)); onFaces(null); } finally { setBusy(false); }
    };

    return (
        <div className="space-y-2 rounded-xl border p-4">
            <div className="text-sm font-medium">{label}</div>
            <input type="file" accept="image/*" disabled={engine.status !== "ready"} onChange={e => pick(e.target.files?.[0])}
                className="block w-full text-sm file:mr-3 file:rounded-md file:border-0 file:bg-zinc-900 file:px-3 file:py-1.5 file:text-white" />
            {/* eslint-disable-next-line @next/next/no-img-element */}
            {url && <img src={url} alt="" className="max-h-56 rounded-md object-contain" />}
            <div className="flex items-center gap-3">
                {busy && <Loader2 className="h-4 w-4 animate-spin" />}
                {faces?.[0]?.aligned && <AlignedThumb data={faces[0].aligned} />}
                {faces?.[0] && (
                    <div className="font-mono text-xs text-muted-foreground">
                        det {faces[0].detScore.toFixed(2)} · sharp {faces[0].quality.sharpness.toFixed(0)}<br />
                        IOD {faces[0].quality.interOcularPx.toFixed(0)} px · {faces[0].timings.totalMs.toFixed(0)} ms
                    </div>
                )}
                {err && <span className="text-sm text-red-600">{err}</span>}
            </div>
        </div>
    );
}

export function ComparePhotos() {
    const engine = useFaceEngine();
    const [a, setA] = useState<WorkerEmbedding[] | null>(null);
    const [b, setB] = useState<WorkerEmbedding[] | null>(null);
    const score = a?.[0] && b?.[0] ? dot(a[0].embedding, b[0].embedding) : null;
    const same = score !== null && score >= DEFAULT_DECISION.threshold;

    return (
        <div className="w-full max-w-3xl space-y-4">
            <div>
                <h1 className="text-2xl font-semibold tracking-tight">Compare two photos</h1>
                <p className="text-sm text-muted-foreground">Runs entirely in your browser: SCRFD → alignment → ArcFace (MobileFaceNet) → cosine similarity.
                    {engine.status === "loading" && " Loading models…"}</p>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
                <Slot label="Photo A" onFaces={setA} />
                <Slot label="Photo B" onFaces={setB} />
            </div>
            {score !== null && (
                <div className={`rounded-xl border p-4 ${same ? "border-green-600" : "border-zinc-300"}`}>
                    <div className="text-lg font-medium">{same ? "Same person" : "Different people"} — similarity {score.toFixed(3)}</div>
                    <p className="text-sm text-muted-foreground">Threshold {DEFAULT_DECISION.threshold}. Same-person pairs typically score 0.5–0.8; different people mostly below 0.25.</p>
                </div>
            )}
        </div>
    );
}
