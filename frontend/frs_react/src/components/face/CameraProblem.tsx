import Link from "next/link";
import { Button } from "@/components/ui/button";

interface CameraProblemProps {
    message: string;
    backHref: string;
    onRetry: () => void;
}

/** A camera or model failure, with a way to retry and a way out. */
export function CameraProblem({ message, backHref, onRetry }: CameraProblemProps) {
    return (
        <div role="alert" className="space-y-3 text-center">
            <p className="font-medium text-red-700">{message}</p>
            <div className="flex flex-col justify-center gap-2 sm:flex-row">
                <Button className="h-11" onClick={onRetry}>Try again</Button>
                <Button asChild variant="outline" className="h-11"><Link href={backHref}>Back</Link></Button>
            </div>
        </div>
    );
}

export const ENGINE_ERROR_MESSAGE = "The face models could not load. Check your connection, then try again.";
