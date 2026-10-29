"use client"

import React, { useCallback, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { CheckCircle2, Loader2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FACE_CONFIG } from "@/face/config";
import { registerEmbeddings } from "@/face/api";
import { QUALITY_MESSAGES } from "@/face/core/quality";
import { FULL_REGISTRATION_PLAN, QUICK_REGISTRATION_PLAN, RegistrationSession } from "@/face/core/registration";
import { POSE_TARGETS } from "@/face/core/pose";
import { FaceCamera } from "@/face/react/FaceCamera";
import { drawOverlay, OverlayTone } from "@/face/react/overlay";
import { personIdFromEmail } from "@/face/react/personId";
import { useCamera } from "@/face/react/useCamera";
import { useFaceEngine } from "@/face/react/useFaceEngine";
import { useVideoFrames } from "@/face/react/useVideoFrames";

type Phase = "capturing" | "uploading" | "done" | "failed";

const POSE_NAMES: Record<string, string> = {
    frontal: "Front",
    turn_left: "Left",
    turn_right: "Right",
    look_up: "Up",
    look_down: "Down",
};

/** Hint towards the target pose, phrased for the mirrored preview (yaw > 0 = the user's own left). */
function poseHint(yaw: number, pitch: number, label: keyof typeof POSE_TARGETS): string | null {
    const t = POSE_TARGETS[label];
    if (yaw < t.yaw[0]) return "Turn a little to your left";
    if (yaw > t.yaw[1]) return "Turn a little to your right";
    if (pitch < t.pitch[0]) return "Lift your chin a little";
    if (pitch > t.pitch[1]) return "Lower your chin a little";
    return "Keep your head straight";
}

const FaceRegistrationForm = () => {
    const router = useRouter();
    const searchParams = useSearchParams();
    const videoRef = useRef<HTMLVideoElement>(null);
    const overlayRef = useRef<HTMLCanvasElement>(null);

    const { engine, error: engineError } = useFaceEngine();
    const { ready: cameraReady, error: cameraError } = useCamera(videoRef);

    const plan = searchParams.get("plan") === "quick" ? QUICK_REGISTRATION_PLAN : FULL_REGISTRATION_PLAN;
    const session = useMemo(() => new RegistrationSession(plan), [plan]);
    const lastEmbedAt = useRef(0);

    const [phase, setPhase] = useState<Phase>("capturing");
    const [message, setMessage] = useState("Initializing camera...");
    const [progress, setProgress] = useState(session.progress);
    const [error, setError] = useState<string | null>(null);

    const email = searchParams.get("email") ?? "";
    const fullname = searchParams.get("fullname") ?? "";

    const submit = useCallback(async () => {
        setPhase("uploading");
        setMessage("Saving your face profile...");
        try {
            await registerEmbeddings(
                { person_id: personIdFromEmail(email || fullname), name: fullname, email },
                session.collected.map(s => ({ embedding: s.embedding, poseLabel: s.poseLabel, pose: s.pose, quality: s.quality })),
            );
            setPhase("done");
            setTimeout(() => router.push("/login"), 3000);
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
            setPhase("failed");
        }
    }, [email, fullname, router, session]);

    useVideoFrames(videoRef, phase === "capturing" && !!engine && cameraReady, (video, ts) => {
        if (!engine) return;
        const w = video.videoWidth, h = video.videoHeight;
        const face = engine.detect(video, ts);
        const target = session.currentTarget;
        let tone: OverlayTone = "neutral";

        if (!face || !target) {
            setMessage(face ? "Done" : "No face detected");
        } else if (!session.wants(face.pose, ts)) {
            tone = "warn";
            const hint = poseHint(face.pose.yaw, face.pose.pitch, target.label);
            setMessage(`${target.instruction}${hint ? ` - ${hint}` : ""}`);
        } else {
            const aligned = engine.align(video, face, w, h);
            if (!aligned.quality.ok) {
                tone = "warn";
                setMessage(QUALITY_MESSAGES[aligned.quality.issues[0]]);
            } else {
                tone = "good";
                setMessage(`${target.instruction} - hold still`);
                const minGap = 1000 / FACE_CONFIG.maxEmbeddingsPerSecond;
                if (!engine.embedderBusy && ts - lastEmbedAt.current >= minGap) {
                    lastEmbedAt.current = ts;
                    const pose = face.pose;
                    const quality = aligned.quality.score;
                    void engine.embed(aligned).then(({ embedding }) => {
                        const result = session.offer({ embedding, pose, quality, timestamp: performance.now() });
                        if (result === "inconsistent") {
                            setMessage("Only one person please");
                        }
                        if (result === "accepted" || result === "complete") {
                            setProgress(session.progress);
                        }
                        if (result === "complete") {
                            void submit();
                        }
                    }).catch(e => console.error("embedding failed", e));
                }
            }
        }
        if (overlayRef.current) {
            drawOverlay(overlayRef.current, w, h, face, tone, session.fraction);
        }
    });

    const fatal = engineError || cameraError;

    return (
        <Card className="w-full max-w-xl">
            <CardContent className="space-y-4 p-6">
                <div>
                    <h2 className="text-xl font-semibold">Face registration</h2>
                    <p className="text-sm text-muted-foreground">
                        {fullname ? `${fullname}, ` : ""}we capture a few angles of your face. Only numeric face
                        embeddings are sent to the server, never images.
                    </p>
                </div>

                <FaceCamera ref={videoRef} overlayRef={overlayRef} className="aspect-[4/3] w-full">
                    {(!engine || !cameraReady) && !fatal && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 text-white">
                            <Loader2 className="h-8 w-8 animate-spin" />
                            <p className="mt-2 text-sm">{engine ? "Starting camera..." : "Loading face models..."}</p>
                        </div>
                    )}
                    {phase === "done" && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 text-white">
                            <CheckCircle2 className="h-12 w-12 text-green-400" />
                            <p className="mt-2 text-lg font-semibold">Face enrolled successfully!</p>
                            <p className="text-sm">Redirecting to login...</p>
                        </div>
                    )}
                </FaceCamera>

                <p className="min-h-6 text-center font-medium" aria-live="polite">
                    {fatal ? <span className="text-red-600">{fatal}</span> : message}
                </p>

                <ol className="flex justify-center gap-2">
                    {progress.map(p => (
                        <li key={p.label}
                            className={`rounded-full px-3 py-1 text-xs ${p.have >= p.need ? "bg-green-600 text-white" : session.currentTarget?.label === p.label ? "bg-blue-600 text-white" : "bg-muted"}`}>
                            {POSE_NAMES[p.label]} {p.have}/{p.need}
                        </li>
                    ))}
                </ol>

                {phase === "failed" && (
                    <div className="space-y-2 text-center">
                        <p className="text-sm text-red-600">Registration failed: {error}</p>
                        <Button onClick={submit}>Retry upload</Button>
                    </div>
                )}
            </CardContent>
        </Card>
    );
};

export default FaceRegistrationForm;
