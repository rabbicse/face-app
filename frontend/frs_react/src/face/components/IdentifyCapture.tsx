"use client";

import { useCallback, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { CheckCircle2 } from "lucide-react";
import { FaceCapture } from "./FaceCapture";
import { IdentifyMode } from "../runtime/modes";
import { FaceCaptureController } from "../runtime/controller";
import { getRepository, IdentifyResponse, PersonInfo } from "../repository";
import { Button } from "@/components/ui/button";

export function IdentifyCapture({ showStats = false, redirectTo = "/dashboard" }: { showStats?: boolean; redirectTo?: string | null }) {
    const router = useRouter();
    const ctrl = useRef<FaceCaptureController | null>(null);
    const [match, setMatch] = useState<{ person: PersonInfo; score: number } | null>(null);
    const [failures, setFailures] = useState(0);

    const createMode = useCallback(() => new IdentifyMode(
        e => getRepository().identify(e),
        (r: IdentifyResponse) => {
            if (r.decision.status === "match") {
                const { meta, score } = r.decision.candidate;
                const person = meta ?? { name: "Unknown" };
                setMatch({ person, score });
                if (redirectTo) {
                    const q = new URLSearchParams({ name: String(person.name ?? ""), email: String(person.email ?? "") });
                    setTimeout(() => router.push(`${redirectTo}?${q}`), 1500);
                }
                return "stop";
            }
            setFailures(f => f + 1);
            return "continue";
        }), [redirectTo, router]);

    return (
        <div className="w-full max-w-2xl space-y-4">
            <div>
                <h1 className="text-2xl font-semibold tracking-tight">Face login</h1>
                <p className="text-sm text-muted-foreground">Just look at the camera — no need to turn your head.</p>
            </div>
            {!match && <FaceCapture createMode={createMode} onController={c => (ctrl.current = c)} showStats={showStats} />}
            {match && (
                <div className="space-y-3 rounded-xl border border-green-600 p-6">
                    <div className="flex items-center gap-2 text-lg font-medium text-green-700">
                        <CheckCircle2 className="h-6 w-6" /> Welcome back, {String(match.person.name)}
                    </div>
                    <p className="text-sm text-muted-foreground">Similarity {match.score.toFixed(3)}{redirectTo ? " — redirecting…" : ""}</p>
                    {!redirectTo && <Button onClick={() => setMatch(null)}>Identify again</Button>}
                </div>
            )}
            {failures >= 3 && !match && (
                <p className="text-sm text-muted-foreground">
                    Still not recognised? <Link className="underline" href="/registration">Register your face</Link> first, or improve the lighting.
                </p>
            )}
        </div>
    );
}
