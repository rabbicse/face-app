import Link from "next/link";
import { CameraOff, RotateCw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface CameraProblemProps {
    message: string;
    backHref: string;
    onRetry: () => void;
}

/** A camera or model failure, with a way to retry and a way out. */
export function CameraProblem({ message, backHref, onRetry }: CameraProblemProps) {
    return (
        <div role="alert" className="flex flex-col items-center gap-4 rounded-xl bg-destructive-soft px-4 py-8 text-center">
            <span className="grid size-12 place-items-center rounded-full bg-card text-destructive shadow-[0_1px_2px_hsl(var(--ink)/0.08)]">
                <CameraOff className="size-6" aria-hidden="true" />
            </span>
            <p className="max-w-sm font-medium text-destructive">{message}</p>
            <div className="flex w-full flex-col justify-center gap-2 sm:w-auto sm:flex-row">
                <Button className="h-11" onClick={onRetry}><RotateCw aria-hidden="true" />Try again</Button>
                <Button asChild variant="outline" className="h-11"><Link href={backHref}>Back</Link></Button>
            </div>
        </div>
    );
}

export const ENGINE_ERROR_MESSAGE = "The face models could not load. Check your connection, then try again.";
