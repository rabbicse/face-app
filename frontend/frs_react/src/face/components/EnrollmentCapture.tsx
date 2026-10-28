"use client";

import { useCallback, useRef, useState, useSyncExternalStore } from "react";
import Link from "next/link";
import { Check, CheckCircle2, Loader2, XCircle } from "lucide-react";
import { FaceCapture } from "./FaceCapture";
import { EnrollmentMode } from "../runtime/modes";
import { FaceCaptureController } from "../runtime/controller";
import { getRepository, PersonInfo } from "../repository";
import type { EnrollmentSample, PoseStepId } from "../core/enrollment";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const STEP_LABEL: Record<string, string> = { front: "Front", left: "Left", right: "Right", up: "Up", down: "Down" };
const noop = () => () => { };

type SaveState = { kind: "capturing" } | { kind: "saving" } | { kind: "saved"; personId: string; templates: number } | { kind: "error"; message: string; templates: EnrollmentSample[] };

export function EnrollmentCapture({ person, showStats = false, steps: stepIds }: { person: PersonInfo; showStats?: boolean; steps?: PoseStepId[] }) {
    const modeRef = useRef<EnrollmentMode | null>(null);
    const [controller, setController] = useState<FaceCaptureController | null>(null);
    const [save, setSave] = useState<SaveState>({ kind: "capturing" });

    const persist = useCallback(async (templates: EnrollmentSample[]) => {
        setSave({ kind: "saving" });
        try {
            const r = await getRepository().register(person, templates);
            setSave({ kind: "saved", personId: r.personId, templates: r.templates });
        } catch (e) {
            setSave({ kind: "error", message: e instanceof Error ? e.message : String(e), templates });
        }
    }, [person]);

    const createMode = useCallback(() => {
        modeRef.current = new EnrollmentMode(t => void persist(t), stepIds);
        return modeRef.current;
    }, [persist, stepIds]);

    // re-render when guidance/progress changes, then read step state from the session
    useSyncExternalStore(controller ? controller.state.subscribe : noop, controller ? controller.state.get : () => null, () => null);
    const progress = modeRef.current?.session.progress();
    const steps = modeRef.current?.session.config.steps ?? [];

    return (
        <div className="w-full max-w-2xl space-y-4">
            <div>
                <h1 className="text-2xl font-semibold tracking-tight">Register your face</h1>
                <p className="text-sm text-muted-foreground">
                    {person.name ? <>Hi <b>{person.name}</b>. </> : null}
                    We capture your face from {stepIds?.length ? new Set(["front", ...stepIds]).size : 5} angles so you are recognised later from any natural angle.
                    Only a 512-number vector per angle is stored{getRepository().kind === "local" ? " — in this browser" : ""}; no photos leave your device.
                </p>
            </div>

            <ol className="flex flex-wrap gap-2">
                {steps.map((s, i) => {
                    const done = progress?.completedSteps.includes(s.id);
                    const active = progress?.stepIndex === i && save.kind === "capturing";
                    return (
                        <li key={s.id} className={cn("flex items-center gap-1 rounded-full border px-3 py-1 text-sm",
                            done && "border-green-600 bg-green-50 text-green-700",
                            active && "border-zinc-900 bg-zinc-900 text-white")}>
                            {done ? <Check className="h-3.5 w-3.5" /> : <span className="text-xs opacity-60">{i + 1}</span>}
                            {STEP_LABEL[s.id] ?? s.id}
                        </li>
                    );
                })}
            </ol>

            {save.kind === "capturing" && (
                <>
                    <FaceCapture createMode={createMode} onController={setController} showStats={showStats} />
                    <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Tip: even lighting on your face, no strong backlight.</span>
                        {progress && progress.stepIndex > 0 && (
                            <button className="underline" onClick={() => modeRef.current?.skip()}>Skip this angle</button>
                        )}
                    </div>
                </>
            )}

            {save.kind === "saving" && (
                <div className="flex items-center gap-2 rounded-xl border p-6"><Loader2 className="h-5 w-5 animate-spin" /> Saving face templates…</div>
            )}
            {save.kind === "saved" && (
                <div className="space-y-4 rounded-xl border border-green-600 p-6">
                    <div className="flex items-center gap-2 text-lg font-medium text-green-700"><CheckCircle2 className="h-6 w-6" /> Face registered</div>
                    <p className="text-sm text-muted-foreground">{save.templates} pose templates stored (id {save.personId.slice(0, 8)}…).</p>
                    <Button asChild><Link href="/face/login">Try face login</Link></Button>
                </div>
            )}
            {save.kind === "error" && (
                <div className="space-y-4 rounded-xl border border-red-500 p-6">
                    <div className="flex items-center gap-2 font-medium text-red-600"><XCircle className="h-5 w-5" /> Could not save: {save.message}</div>
                    <div className="flex gap-2">
                        <Button onClick={() => void persist(save.templates)}>Retry saving</Button>
                        <Button variant="outline" onClick={() => setSave({ kind: "capturing" })}>Capture again</Button>
                    </div>
                </div>
            )}
        </div>
    );
}
