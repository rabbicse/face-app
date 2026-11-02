import type { Metadata } from "next";
import { Suspense } from "react";
import FaceRegistrationForm from "@/components/face/FaceRegistrationForm";

export const metadata: Metadata = { title: "Scan your face" };

export default function FaceRegistrationPage() {
    return (
        <main className="flex min-h-screen w-full items-center justify-center px-4 py-8">
            {/* useSearchParams needs a Suspense boundary for static rendering */}
            <Suspense>
                <FaceRegistrationForm />
            </Suspense>
        </main>
    );
}
