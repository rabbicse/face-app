import type { Metadata } from "next";
import { Suspense } from "react";
import FaceRegistrationForm from "@/components/face/FaceRegistrationForm";

export const metadata: Metadata = { title: "Scan your face" };

export default function FaceRegistrationPage() {
    return (
        <main className="flex w-full flex-1 items-start justify-center px-4 pb-16 pt-4 sm:items-center sm:pt-8">
            {/* useSearchParams needs a Suspense boundary for static rendering */}
            <Suspense>
                <FaceRegistrationForm />
            </Suspense>
        </main>
    );
}
