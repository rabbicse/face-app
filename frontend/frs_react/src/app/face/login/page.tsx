"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { IdentifyCapture } from "@/face/components/IdentifyCapture";

function Login() {
    const q = useSearchParams();
    return <IdentifyCapture showStats={q.has("debug")} redirectTo={q.get("redirect") === "0" ? null : "/dashboard"} />;
}

export default function FaceLoginPage() {
    return (
        <main className="flex min-h-screen w-full items-start justify-center px-4 py-8">
            <Suspense>
                <Login />
            </Suspense>
        </main>
    );
}
