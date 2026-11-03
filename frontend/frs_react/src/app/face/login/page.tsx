import type { Metadata } from "next";
import FaceLoginForm from "@/components/face/FaceLoginForm";

export const metadata: Metadata = { title: "Sign in with your face" };

export default function FaceLoginPage() {
    return (
        <main className="flex w-full flex-1 items-start justify-center px-4 pb-16 pt-4 sm:items-center sm:pt-8">
            <FaceLoginForm />
        </main>
    );
}
