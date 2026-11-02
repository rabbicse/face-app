/** Turns a getUserMedia failure into what happened and what to do next. */
export function describeCameraError(error: unknown): string {
    switch (error instanceof Error ? error.name : "") {
        case "NotAllowedError":
        case "SecurityError":
            return "Camera access is blocked. Allow the camera in your browser's site settings, then try again.";
        case "NotFoundError":
        case "OverconstrainedError":
            return "No camera was found. Connect a camera, then try again.";
        case "NotReadableError":
        case "AbortError":
            return "Your camera is being used by another app. Close that app, then try again.";
        case "InsecureContextError":
            return "This browser only allows the camera on a secure page. Open this site over https or on localhost.";
        default:
            return "The camera could not start. Check that it is connected, then try again.";
    }
}
