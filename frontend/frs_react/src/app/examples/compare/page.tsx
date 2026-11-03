"use client"

import { useState } from "react";
import { UserCheck, Users } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { dot } from "@/face/core/embedding";
import { useFaceEngine } from "@/face/react/useFaceEngine";
import type { FaceEngine } from "@/face/runtime/engine";

/** Same default as the backend (FRS_MATCH_THRESHOLD). */
const THRESHOLD = 0.45;

async function embedFile(engine: FaceEngine, file: File): Promise<{ embedding: Float32Array; preview: string }> {
    const bitmap = await createImageBitmap(file);
    try {
        const face = await engine.detectImage(bitmap, bitmap.width, bitmap.height);
        if (!face) throw new Error(`no face found in ${file.name}`);
        const aligned = engine.align(bitmap, face, bitmap.width, bitmap.height);
        const canvas = document.createElement("canvas");
        canvas.width = canvas.height = aligned.crop.width;
        canvas.getContext("2d")!.putImageData(aligned.crop, 0, 0);
        const preview = canvas.toDataURL();
        const { embedding } = await engine.embed(aligned);
        return { embedding, preview };
    } finally {
        bitmap.close();
    }
}

export default function ComparePage() {
    const { engine, error } = useFaceEngine();
    const [faces, setFaces] = useState<({ embedding: Float32Array; preview: string } | null)[]>([null, null]);
    const [status, setStatus] = useState<string | null>(null);

    const onPick = async (index: number, file?: File) => {
        if (!engine || !file) return;
        setStatus("Computing embedding...");
        try {
            const result = await embedFile(engine, file);
            setFaces(prev => prev.map((f, i) => (i === index ? result : f)));
            setStatus(null);
        } catch (e) {
            setStatus(e instanceof Error ? e.message : String(e));
        }
    };

    const similarity = faces[0] && faces[1] ? dot(faces[0].embedding, faces[1].embedding) : null;

    return (
        <main className="mx-auto w-full max-w-2xl flex-1 space-y-6 px-4 pb-16 pt-6 sm:px-6 sm:pt-10">
            <div className="space-y-2">
                <h1 className="text-3xl font-semibold tracking-tight">Compare two photos</h1>
                <p className="text-muted-foreground">Both photos are analysed in your browser. Nothing is uploaded.</p>
            </div>
            {error && <p role="alert" className="rounded-lg bg-destructive-soft px-3 py-2 font-medium text-destructive">{error}</p>}
            {!engine && !error && <p className="text-muted-foreground">Loading models...</p>}
            <div className="grid grid-cols-2 gap-4">
                {[0, 1].map(i => (
                    <Card key={i}>
                        <CardContent className="space-y-2 p-4">
                            <Label htmlFor={`photo-${i}`}>Photo {i + 1}</Label>
                            <Input id={`photo-${i}`} type="file" accept="image/*" disabled={!engine}
                                onChange={e => onPick(i, e.target.files?.[0])} />
                            {faces[i] && (
                                // eslint-disable-next-line @next/next/no-img-element
                                <img src={faces[i]!.preview} alt={`aligned face ${i + 1}`} width={112} height={112} className="rounded-lg border bg-muted" />
                            )}
                        </CardContent>
                    </Card>
                ))}
            </div>
            {status && <p className="text-muted-foreground">{status}</p>}
            {similarity !== null && (
                <p data-testid="similarity"
                    className={`flex flex-wrap items-center gap-x-3 gap-y-1 rounded-xl px-4 py-3 text-lg ${similarity >= THRESHOLD ? "bg-success-soft text-success" : "bg-muted text-foreground"}`}>
                    {similarity >= THRESHOLD
                        ? <UserCheck className="size-5 shrink-0" aria-hidden="true" />
                        : <Users className="size-5 shrink-0" aria-hidden="true" />}
                    <span>Cosine similarity: <span className="font-mono font-semibold tabular-nums">{similarity.toFixed(3)}</span></span>
                    <span className="font-medium">{similarity >= THRESHOLD ? "- same person" : "- different people"}</span>
                </p>
            )}
        </main>
    );
}
