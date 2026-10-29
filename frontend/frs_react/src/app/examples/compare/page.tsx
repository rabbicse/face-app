"use client"

import { useState } from "react";
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
        <main className="mx-auto max-w-2xl space-y-4 p-6">
            <h1 className="text-2xl font-bold">Compare two photos</h1>
            {error && <p className="text-red-600">{error}</p>}
            {!engine && !error && <p>Loading models...</p>}
            <div className="grid grid-cols-2 gap-4">
                {[0, 1].map(i => (
                    <Card key={i}>
                        <CardContent className="space-y-2 p-4">
                            <Label htmlFor={`photo-${i}`}>Photo {i + 1}</Label>
                            <Input id={`photo-${i}`} type="file" accept="image/*" disabled={!engine}
                                onChange={e => onPick(i, e.target.files?.[0])} />
                            {faces[i] && (
                                // eslint-disable-next-line @next/next/no-img-element
                                <img src={faces[i]!.preview} alt={`aligned face ${i + 1}`} width={112} height={112} className="rounded border" />
                            )}
                        </CardContent>
                    </Card>
                ))}
            </div>
            {status && <p>{status}</p>}
            {similarity !== null && (
                <p className="text-lg" data-testid="similarity">
                    Cosine similarity: <span className="font-mono">{similarity.toFixed(3)}</span>{" "}
                    {similarity >= THRESHOLD ? "- same person" : "- different people"}
                </p>
            )}
        </main>
    );
}
