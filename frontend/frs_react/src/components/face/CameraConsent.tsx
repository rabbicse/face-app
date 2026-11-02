import Link from "next/link";
import { Button } from "@/components/ui/button";

interface CameraConsentProps {
    title: string;
    intro: string;
    points: string[];
    actionLabel: string;
    backHref: string;
    onAccept: () => void;
}

/** Shown before the camera starts, so the browser's permission prompt never arrives unexplained. */
export function CameraConsent({ title, intro, points, actionLabel, backHref, onAccept }: CameraConsentProps) {
    return (
        <div className="space-y-4">
            <div className="space-y-1">
                <h1 className="text-xl font-semibold">{title}</h1>
                <p className="text-muted-foreground">{intro}</p>
            </div>
            <ul className="list-disc space-y-2 pl-5 text-sm">
                {points.map(point => <li key={point}>{point}</li>)}
            </ul>
            <div className="flex flex-col gap-2 sm:flex-row">
                <Button className="h-11 flex-1" onClick={onAccept}>{actionLabel}</Button>
                <Button asChild variant="outline" className="h-11"><Link href={backHref}>Back</Link></Button>
            </div>
        </div>
    );
}
