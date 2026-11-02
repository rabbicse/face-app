"use client"

import React, { useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { CheckCircle2, Loader2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { CameraConsent } from "@/components/face/CameraConsent";
import { CameraProblem, ENGINE_ERROR_MESSAGE } from "@/components/face/CameraProblem";
import { FACE_CONFIG } from "@/face/config";
import { identifyEmbedding, PersonInfo } from "@/face/api";
import { IdentificationAggregator } from "@/face/core/identification";
import { BlinkDetector } from "@/face/core/liveness";
import { QUALITY_MESSAGES } from "@/face/core/quality";
import { FaceCamera } from "@/face/react/FaceCamera";
import { drawOverlay, OverlayTone } from "@/face/react/overlay";
import { saveSignedIn } from "@/face/react/demoSession";
import { useAnnouncement } from "@/face/react/useAnnouncement";
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
    const [cameraOn, setCameraOn] = useState(false);
    const { ready: cameraReady, error: cameraError, retry: retryCamera } = useCamera(videoRef, cameraOn);

    const aggregator = useMemo(() => new IdentificationAggregator(), []);
    const blink = useMemo(() => new BlinkDetector(), []);
    const lastEmbedAt = useRef(0);
    const lastAlignAt = useRef(0);
    const lastTone = useRef<OverlayTone>("neutral");
    const queryInFlight = useRef(false);
    const nextQueryAt = useRef(0);

    const [message, setMessage] = useState("Look at the camera");
    const [person, setPerson] = useState<PersonInfo | null>(null);

    const query = async (embedding: Float32Array) => {
        queryInFlight.current = true;
        try {
            const res = await identifyEmbedding(embedding);
            if (res.status === 0 && res.person) {
                setPerson(res.person);
                saveSignedIn({ personId: res.person.person_id, name: res.person.name, email: res.person.email, score: res.score });
                setTimeout(() => router.push("/dashboard"), 2000);
            } else {
                setMessage(res.reason === "no_enrolled_faces" ? "No faces are registered yet. Create an account first." : "Face not recognized. Keep looking at the camera.");
                nextQueryAt.current = performance.now() + RETRY_DELAY_MS;
            }
        } catch (e) {
            console.warn("identify failed", e);
            setMessage("Can't reach the server. Trying again...");
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
            blink.reset();
            setMessage("No face detected");
        } else {
            if (face.eyesClosed !== undefined) blink.update(face.eyesClosed, ts);
            const live = !FACE_CONFIG.liveness.requireBlink || blink.hasRecentBlink(ts, FACE_CONFIG.liveness.blinkWindowMs);
            // identification has no head pose requirement, only image quality. Warping + the quality check
            // (a canvas readback) only run as often as an embedding may be computed; other frames reuse the verdict.
            const minGap = 1000 / FACE_CONFIG.maxEmbeddingsPerSecond;
            const aligned = ts - lastAlignAt.current >= minGap && !engine.embedderBusy ? engine.align(video, face, w, h) : null;
            if (aligned) lastAlignAt.current = ts;
            if (!aligned) {
                tone = lastTone.current;
            } else if (!aligned.quality.ok) {
                tone = "warn";
                setMessage(QUALITY_MESSAGES[aligned.quality.issues[0]]);
            } else {
                tone = "good";
                if (!live) setMessage("Please blink to continue");
                else if (ts >= nextQueryAt.current) setMessage("Recognizing...");
                const canQuery = !queryInFlight.current && ts >= nextQueryAt.current;
                if (canQuery && ts - lastEmbedAt.current >= minGap) {
                    lastEmbedAt.current = ts;
                    void engine.embed(aligned, FACE_CONFIG.embedding.identificationFlipTTA).then(({ embedding }) => {
                        const fused = aggregator.add(embedding, performance.now());
                        const stillLive = !FACE_CONFIG.liveness.requireBlink
                            || blink.hasRecentBlink(performance.now(), FACE_CONFIG.liveness.blinkWindowMs);
                        if (fused && stillLive && !queryInFlight.current) {
                            void query(fused);
                        }
                    }).catch(e => console.error("embedding failed", e));
                }
            }
        }
        lastTone.current = tone;
        if (overlayRef.current) {
            drawOverlay(overlayRef.current, w, h, face, tone);
        }
    });

    const fatal = engineError ? ENGINE_ERROR_MESSAGE : cameraError;
    const status = person ? "Face match found!" : message;
    const announcement = useAnnouncement(status);

    if (!cameraOn) {
        return (
            <Card className="w-full max-w-xl">
                <CardContent className="p-6">
                    <CameraConsent
                        title="Sign in with your face"
                        intro="Your camera stays off until you turn it on."
                        points={[
                            "Photos and video stay on this device.",
                            "Only a numeric face signature is sent to the server, to find your match.",
                            "On your first visit, this page downloads about 16 MB of face models.",
                        ]}
                        actionLabel="Turn on camera"
                        backHref="/"
                        onAccept={() => setCameraOn(true)}
                    />
                    <p className="mt-4 text-center text-sm text-muted-foreground">
                        Not registered yet? <Link className="text-foreground underline underline-offset-4" href="/registration">Create an account</Link>
                    </p>
                </CardContent>
            </Card>
        );
    }

    return (
        <Card className="w-full max-w-xl">
            <CardContent className="space-y-4 p-6">
                <h1 className="text-xl font-semibold">Sign in with your face</h1>
                {/* stays mounted when hidden: "Try again" restarts the camera on the same video element */}
                <FaceCamera ref={videoRef} overlayRef={overlayRef} className={`aspect-[4/3] w-full ${fatal ? "hidden" : ""}`}>
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
                            <p className="text-sm">Taking you to your page...</p>
                        </div>
                    )}
                </FaceCamera>
                {fatal ? (
                    <CameraProblem message={fatal} backHref="/"
                        onRetry={engineError ? () => window.location.reload() : retryCamera} />
                ) : (
                    <>
                        <p className="min-h-6 text-center font-medium" aria-hidden="true">{status}</p>
                        <p className="sr-only" aria-live="polite">{announcement}</p>
                    </>
                )}
                {!person && (
                    <div className="space-y-1 text-center text-sm text-muted-foreground">
                        <p>Not registered yet? <Link className="text-foreground underline underline-offset-4" href="/registration">Create an account</Link></p>
                        {!fatal && <p><Link className="text-foreground underline underline-offset-4" href="/">Back to start</Link></p>}
                    </div>
                )}
            </CardContent>
        </Card>
    );
};

export default FaceLoginForm;
