"use client"

import { Suspense } from "react";
import FaceRegistrationForm from "@/components/face/FaceRegistrationForm";

export default function FaceRegistrationPage() {
    return (
        <div className="flex min-h-screen w-full items-center justify-center px-4 py-8">
            {/* useSearchParams needs a Suspense boundary for static rendering */}
            <Suspense>
                <FaceRegistrationForm />
            </Suspense>
        </div>
    );
}
