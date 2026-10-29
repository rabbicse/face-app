"use client";

import { usePreloadFaceEngine } from "./useFaceEngine";

/** Renders nothing; starts downloading / compiling the face models while the browser is idle. */
export function FacePreloader() {
    usePreloadFaceEngine();
    return null;
}
