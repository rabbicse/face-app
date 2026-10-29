"use client";

import { forwardRef, ReactNode } from "react";

interface FaceCameraProps {
    overlayRef: React.RefObject<HTMLCanvasElement>;
    children?: ReactNode;
    className?: string;
}

/** Mirrored (selfie) video with an overlay canvas in video pixel coordinates. */
export const FaceCamera = forwardRef<HTMLVideoElement, FaceCameraProps>(function FaceCamera({ overlayRef, children, className }, videoRef) {
    return (
        <div className={`relative overflow-hidden rounded-xl bg-black ${className ?? ""}`}>
            {/* mirroring is done with CSS only; the models always see the original frame */}
            <video ref={videoRef} className="block h-full w-full -scale-x-100 object-cover" muted playsInline />
            <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100 object-cover" />
            {children}
        </div>
    );
});
