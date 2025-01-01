"use client"

import FaceLoginForm from "@/components/face/FaceLoginForm";
import { TensorFlowProvider } from "@/components/tensorflow/TensorflowContext";

export default function FaceLoginPage() {
    return (
        <div className="flex h-screen w-full items-center justify-center px-4">
            <TensorFlowProvider>
                <FaceLoginForm />
            </TensorFlowProvider>
        </div>
    );
}
