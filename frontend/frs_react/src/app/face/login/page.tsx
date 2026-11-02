import type { Metadata } from "next";
import FaceLoginForm from "@/components/face/FaceLoginForm";

export const metadata: Metadata = { title: "Sign in with your face" };

export default function FaceLoginPage() {
    return (
        <main className="flex min-h-screen w-full items-center justify-center px-4 py-8">
            <FaceLoginForm />
        </main>
    );
}
