"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { useMemo } from "react";
import { EnrollmentCapture } from "@/face/components/EnrollmentCapture";
import type { PoseStepId } from "@/face/core/enrollment";

function Registration() {
    const q = useSearchParams();
    // ?steps=front,left,right limits the pose steps (front is always included)
    const stepsParam = q.get("steps");
    const steps = useMemo(() => stepsParam?.split(",").filter(Boolean) as PoseStepId[] | undefined, [stepsParam]);
    return (
        <EnrollmentCapture
            person={{ name: q.get("fullname") ?? "", email: q.get("email") ?? "" }}
            showStats={q.has("debug")}
            steps={steps}
        />
    );
}

export default function FaceRegistrationPage() {
    return (
        <main className="flex min-h-screen w-full items-start justify-center px-4 py-8">
            <Suspense>
                <Registration />
            </Suspense>
        </main>
    );
}
