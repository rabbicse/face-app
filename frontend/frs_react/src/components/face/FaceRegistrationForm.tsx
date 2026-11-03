"use client"

import React, { useCallback, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Check } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { CameraConsent } from "@/components/face/CameraConsent";
import { CameraProblem, ENGINE_ERROR_MESSAGE } from "@/components/face/CameraProblem";
import { CameraLoading, CameraPreparing, CameraSuccess, ScanStatus, ScanTone } from "@/components/face/ScanUi";
import { cn } from "@/lib/utils";
import { FACE_CONFIG } from "@/face/config";
import { registerEmbeddings } from "@/face/api";
import { QUALITY_MESSAGES } from "@/face/core/quality";
import { FULL_REGISTRATION_PLAN, QUICK_REGISTRATION_PLAN, RegistrationSession } from "@/face/core/registration";
import { POSE_TARGETS } from "@/face/core/pose";
import { FaceCamera } from "@/face/react/FaceCamera";
import { drawOverlay, OverlayTone } from "@/face/react/overlay";
import { clearPendingEnrolment, readPendingEnrolment } from "@/face/react/demoSession";
import { personIdFromEmail } from "@/face/react/personId";
import { useAnnouncement } from "@/face/react/useAnnouncement";
import { useCamera } from "@/face/react/useCamera";
import { useHydrated } from "@/face/react/useHydrated";
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
    const [cameraOn, setCameraOn] = useState(false);
    const { ready: cameraReady, error: cameraError, retry: retryCamera } = useCamera(videoRef, cameraOn);

    const plan = searchParams.get("plan") === "quick" ? QUICK_REGISTRATION_PLAN : FULL_REGISTRATION_PLAN;
    const session = useMemo(() => new RegistrationSession(plan), [plan]);
    const lastEmbedAt = useRef(0);
    const lastTone = useRef<OverlayTone>("neutral");

    const [phase, setPhase] = useState<Phase>("capturing");
    const [message, setMessage] = useState("Initializing camera...");
    const [statusTone, setStatusTone] = useState<ScanTone>("idle");
    const [progress, setProgress] = useState(session.progress);
    const [uploadFailed, setUploadFailed] = useState(false);

    // The registration form leaves these in sessionStorage. URL parameters only serve the demo link on /examples.
    const identityLoaded = useHydrated();
    const identity = useMemo(() => {
        if (!identityLoaded) return null;
        const fromUrl = { fullname: searchParams.get("fullname") ?? "", email: searchParams.get("email") ?? "" };
        return fromUrl.email || fromUrl.fullname ? fromUrl : readPendingEnrolment();
    }, [identityLoaded, searchParams]);
    const email = identity?.email ?? "";
    const fullname = identity?.fullname ?? "";

    const submit = useCallback(async () => {
        setPhase("uploading");
        setUploadFailed(false);
        setMessage("Saving your face profile...");
        setStatusTone("working");
        try {
            await registerEmbeddings(
                { person_id: personIdFromEmail(email || fullname), name: fullname, email },
                session.collected.map(s => ({ embedding: s.embedding, poseLabel: s.poseLabel, pose: s.pose, quality: s.quality })),
            );
            clearPendingEnrolment();
            setPhase("done");
            setTimeout(() => router.push("/face/login"), 3000);
        } catch (e) {
            console.warn("registration upload failed", e);
            setUploadFailed(true);
            setMessage("Your face profile was not saved");
            setStatusTone("error");
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
            setStatusTone(face ? "success" : "idle");
        } else if (!session.wants(face.pose, ts)) {
            tone = "warn";
            const rel = session.relativePose(face.pose);
            const hint = poseHint(rel.yaw, rel.pitch, target.label);
            setMessage(`${target.instruction}${hint ? `. ${hint}.` : ""}`);
            setStatusTone("working");
        } else if (engine.embedderBusy || ts - lastEmbedAt.current < 1000 / FACE_CONFIG.maxEmbeddingsPerSecond) {
            // right pose, but no embedding is due on this frame: skip the warp + readback, keep the last verdict
            tone = lastTone.current === "warn" ? "warn" : "good";
        } else {
            lastEmbedAt.current = ts;
            const aligned = engine.align(video, face, w, h);
            if (!aligned.quality.ok) {
                tone = "warn";
                setMessage(QUALITY_MESSAGES[aligned.quality.issues[0]]);
                setStatusTone("warn");
            } else {
                tone = "good";
                setMessage(`${target.instruction}. Hold still.`);
                setStatusTone("working");
                const pose = face.pose;
                const quality = aligned.quality.score;
                void engine.embed(aligned).then(({ embedding }) => {
                    const result = session.offer({ embedding, pose, quality, timestamp: performance.now() });
                    if (result === "inconsistent") {
                        setMessage("Only one person please");
                        setStatusTone("warn");
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
        lastTone.current = tone;
        if (overlayRef.current) {
            drawOverlay(overlayRef.current, w, h, face, tone, {
                progress: session.fraction,
                axes: true,
                aim: face && target ? { current: session.relativePose(face.pose), target: POSE_TARGETS[target.label] } : undefined,
            });
        }
    });

    const fatal = engineError ? ENGINE_ERROR_MESSAGE : cameraError;
    const announcement = useAnnouncement(message);

    if (!identityLoaded) return null;

    if (!identity) {
        return (
            <Card className="w-full max-w-xl">
                <CardContent className="space-y-5 p-5 sm:p-8">
                    <h1 className="text-2xl font-semibold tracking-tight">Start with your name and email</h1>
                    <p className="text-muted-foreground">Your face profile needs a name and email before the scan.</p>
                    <Button asChild className="h-11 w-full"><Link href="/registration">Create an account</Link></Button>
                </CardContent>
            </Card>
        );
    }

    if (!cameraOn) {
        return (
            <Card className="w-full max-w-xl">
                <CardContent className="p-6 sm:p-8">
                    <CameraConsent
                        title="Before we scan your face"
                        intro="Your camera stays off until you agree. You'll hold still, then turn your head slightly in a few directions."
                        points={[
                            "Photos and video stay on this device.",
                            "Numeric face signatures, your name and your email are sent to the server. Face signatures are biometric data.",
                            "They are kept until you delete them. You can delete them after you sign in.",
                            "On your first visit, this page downloads about 16 MB of face models.",
                        ]}
                        actionLabel="I agree, turn on camera"
                        backHref="/registration"
                        onAccept={() => setCameraOn(true)}
                    />
                </CardContent>
            </Card>
        );
    }

    return (
        <Card className="w-full max-w-xl">
            <CardContent className="space-y-5 p-5 sm:p-8">
                <h1 className="text-2xl font-semibold tracking-tight">Scan your face</h1>

                {/* stays mounted when hidden: "Try again" restarts the camera on the same video element */}
                <FaceCamera ref={videoRef} overlayRef={overlayRef} className={`aspect-[4/3] w-full ${fatal ? "hidden" : ""}`}>
                    {!cameraReady && !fatal && <CameraLoading label="Starting camera..." />}
                    {cameraReady && !engine && !fatal && <CameraPreparing />}
                    {phase === "done" && <CameraSuccess title="Face enrolled successfully!" detail="Taking you to sign in..." />}
                </FaceCamera>

                {fatal ? (
                    <CameraProblem message={fatal} backHref="/registration"
                        onRetry={engineError ? () => window.location.reload() : retryCamera} />
                ) : (
                    <ScanStatus message={phase === "done" ? "Your face profile is saved" : message}
                        tone={phase === "done" ? "success" : statusTone} announcement={announcement} />
                )}

                <ol className="flex flex-wrap justify-center gap-2" aria-label="Scan progress">
                    {progress.map(p => {
                        const done = p.have >= p.need;
                        const current = !done && session.currentTarget?.label === p.label;
                        return (
                            <li key={p.label} aria-current={current ? "step" : undefined}
                                className={cn("inline-flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-medium tabular-nums transition-colors duration-300",
                                    done && "bg-success-soft text-success",
                                    current && "bg-primary text-primary-foreground shadow-[0_2px_8px_-2px_hsl(var(--primary)/0.6)]",
                                    !done && !current && "bg-muted text-muted-foreground")}>
                                {done && <Check className="size-3.5" strokeWidth={3} aria-hidden="true" />}
                                {POSE_NAMES[p.label]} {p.have}/{p.need}
                            </li>
                        );
                    })}
                </ol>

                {phase === "failed" && uploadFailed && (
                    <div role="alert" className="space-y-2 text-center">
                        <p className="text-sm font-medium text-destructive">
                            We couldn&apos;t reach the server. Your scan is still here, so you won&apos;t need to repeat it.
                        </p>
                        <Button className="h-11" onClick={submit}>Try saving again</Button>
                    </div>
                )}

                {phase !== "done" && !fatal && (
                    <p className="text-center text-sm">
                        <Link className="link font-normal text-muted-foreground decoration-muted-foreground/40" href="/registration">Cancel and go back</Link>
                    </p>
                )}
            </CardContent>
        </Card>
    );
};

export default FaceRegistrationForm;
