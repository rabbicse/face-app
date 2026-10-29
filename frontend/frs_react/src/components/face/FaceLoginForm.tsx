"use client"

import React, { useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { CheckCircle2, Loader2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { FACE_CONFIG } from "@/face/config";
import { identifyEmbedding, PersonInfo } from "@/face/api";
import { IdentificationAggregator } from "@/face/core/identification";
import { QUALITY_MESSAGES } from "@/face/core/quality";
import { FaceCamera } from "@/face/react/FaceCamera";
import { drawOverlay, OverlayTone } from "@/face/react/overlay";
import { useCamera } from "@/face/react/useCamera";
import { useFaceEngine } from "@/face/react/useFaceEngine";
import { useVideoFrames } from "@/face/react/useVideoFrames";

/** Pause between two server queries after a miss, so a stranger cannot flood the API. */
const RETRY_DELAY_MS = 700;

const FaceLoginForm = () => {
    const router = useRouter();
    const videoRef = useRef<HTMLVideoElement>(null);
    const overlayRef = useRef<HTMLCanvasElement>(null);

    const { engine, error: engineError } = useFaceEngine();
    const { ready: cameraReady, error: cameraError } = useCamera(videoRef);

    const aggregator = useMemo(() => new IdentificationAggregator(), []);
    const lastEmbedAt = useRef(0);
    const queryInFlight = useRef(false);
    const nextQueryAt = useRef(0);

    const [message, setMessage] = useState("Look at the camera");
    const [attempts, setAttempts] = useState(0);
    const [person, setPerson] = useState<PersonInfo | null>(null);

    const query = async (embedding: Float32Array) => {
        queryInFlight.current = true;
        try {
            const res = await identifyEmbedding(embedding);
            if (res.status === 0 && res.person) {
                setPerson(res.person);
                const params = new URLSearchParams({ name: res.person.name, email: res.person.email ?? "" });
                setTimeout(() => router.push(`/dashboard?${params}`), 2000);
            } else {
                setAttempts(a => a + 1);
                setMessage(res.reason === "no_enrolled_faces" ? "No faces registered yet" : "Face not recognized, keep looking at the camera");
                nextQueryAt.current = performance.now() + RETRY_DELAY_MS;
            }
        } catch (e) {
            setMessage(`Server error: ${e instanceof Error ? e.message : String(e)}`);
            nextQueryAt.current = performance.now() + 3 * RETRY_DELAY_MS;
        } finally {
            queryInFlight.current = false;
        }
    };

    useVideoFrames(videoRef, !person && !!engine && cameraReady, (video, ts) => {
        if (!engine) return;
        const w = video.videoWidth, h = video.videoHeight;
        const face = engine.detect(video, ts);
        let tone: OverlayTone = "neutral";

        if (!face) {
            aggregator.reset();
            setMessage("No face detected");
        } else {
            // identification has no head pose requirement, only image quality
            const aligned = engine.align(video, face, w, h);
            if (!aligned.quality.ok) {
                tone = "warn";
                setMessage(QUALITY_MESSAGES[aligned.quality.issues[0]]);
            } else {
                tone = "good";
                const minGap = 1000 / FACE_CONFIG.maxEmbeddingsPerSecond;
                const canQuery = !queryInFlight.current && ts >= nextQueryAt.current;
                if (canQuery && !engine.embedderBusy && ts - lastEmbedAt.current >= minGap) {
                    lastEmbedAt.current = ts;
                    void engine.embed(aligned).then(({ embedding }) => {
                        const fused = aggregator.add(embedding, performance.now());
                        if (fused && !queryInFlight.current) {
                            void query(fused);
                        }
                    }).catch(e => console.error("embedding failed", e));
                }
            }
        }
        if (overlayRef.current) {
            drawOverlay(overlayRef.current, w, h, face, tone);
        }
    });

    const fatal = engineError || cameraError;

    return (
        <Card className="w-full max-w-xl">
            <CardContent className="space-y-4 p-6">
                <h2 className="text-xl font-semibold">Login with your face</h2>
                <FaceCamera ref={videoRef} overlayRef={overlayRef} className="aspect-[4/3] w-full">
                    {(!engine || !cameraReady) && !fatal && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 text-white">
                            <Loader2 className="h-8 w-8 animate-spin" />
                            <p className="mt-2 text-sm">{engine ? "Starting camera..." : "Loading face models..."}</p>
                        </div>
                    )}
                    {person && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 text-white">
                            <CheckCircle2 className="h-12 w-12 text-green-400" />
                            <h2 className="mt-2 text-xl font-bold">Welcome, {person.name}!</h2>
                            <p className="text-sm">Redirecting...</p>
                        </div>
                    )}
                </FaceCamera>
                <p className="min-h-6 text-center font-medium" aria-live="polite">
                    {fatal ? <span className="text-red-600">{fatal}</span> : person ? "Face match found!" : message}
                </p>
                {attempts >= 5 && !person && (
                    <p className="text-center text-sm text-muted-foreground">
                        Not registered yet? <Link className="underline" href="/registration">Create an account</Link>
                    </p>
                )}
            </CardContent>
        </Card>
    );
};

export default FaceLoginForm;
