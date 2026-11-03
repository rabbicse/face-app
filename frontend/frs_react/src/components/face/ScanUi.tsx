import type { ReactNode } from "react";
import { AlertTriangle, CheckCircle2, Loader2, ScanFace, UserRoundSearch } from "lucide-react";
import { cn } from "@/lib/utils";

export type ScanTone = "idle" | "working" | "warn" | "success" | "error";

const TONE_STYLE: Record<ScanTone, string> = {
    idle: "bg-muted text-foreground",
    working: "bg-primary-soft text-primary-soft-foreground",
    warn: "bg-warning-soft text-warning",
    success: "bg-success-soft text-success",
    error: "bg-destructive-soft text-destructive",
};

const TONE_ICON: Record<ScanTone, typeof ScanFace> = {
    idle: UserRoundSearch,
    working: ScanFace,
    warn: AlertTriangle,
    success: CheckCircle2,
    error: AlertTriangle,
};

/**
 * The live instruction under the camera. The visible line changes with every frame, so it is hidden from
 * screen readers; `announcement` is the settled version, read out politely.
 */
export function ScanStatus({ message, tone, announcement }: { message: string; tone: ScanTone; announcement: string }) {
    const Icon = TONE_ICON[tone];
    return (
        <>
            <p aria-hidden="true"
                className={cn("flex min-h-11 items-center justify-center gap-2 rounded-xl px-4 py-2 text-center font-medium transition-colors duration-300", TONE_STYLE[tone])}>
                <Icon className="size-[18px] shrink-0" aria-hidden="true" />
                <span>{message}</span>
            </p>
            <p className="sr-only" aria-live="polite">{announcement}</p>
        </>
    );
}

/** Ink-tinted layer over the camera preview for loading and success moments. */
export function CameraScrim({ children }: { children: ReactNode }) {
    return (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-ink/70 px-6 text-center text-white backdrop-blur-[2px]">
            {children}
        </div>
    );
}

export function CameraLoading({ label }: { label: string }) {
    return (
        <CameraScrim>
            <Loader2 className="size-8 animate-spin text-[hsl(186_70%_70%)]" aria-hidden="true" />
            <p className="text-sm font-medium">{label}</p>
        </CameraScrim>
    );
}

/** Over the live preview while the face models finish loading: the user already sees themselves. */
export function CameraPreparing() {
    return (
        <div className="absolute inset-x-0 bottom-3 flex justify-center px-3">
            <p className="inline-flex items-center gap-2 rounded-full bg-ink/75 px-3.5 py-1.5 text-sm font-medium text-white backdrop-blur-sm">
                <Loader2 className="size-4 animate-spin text-[hsl(186_70%_70%)]" aria-hidden="true" />
                Preparing face recognition...
            </p>
        </div>
    );
}

export function CameraSuccess({ title, detail }: { title: string; detail: string }) {
    return (
        <CameraScrim>
            <span className="grid size-16 place-items-center rounded-full bg-success text-success-foreground shadow-[0_8px_24px_-8px_hsl(155_70%_27%/0.8)] animate-in zoom-in-50 fade-in duration-300 ease-out">
                <CheckCircle2 className="size-9" aria-hidden="true" />
            </span>
            <h2 className="mt-2 text-xl font-semibold">{title}</h2>
            <p className="text-sm text-white/85">{detail}</p>
        </CameraScrim>
    );
}
