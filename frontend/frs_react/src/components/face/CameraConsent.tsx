import Link from "next/link";
import { Camera, Check } from "lucide-react";
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
        <div className="space-y-6">
            <div className="flex items-start gap-4">
                <span className="grid size-12 shrink-0 place-items-center rounded-2xl bg-primary-soft text-primary">
                    <Camera className="size-6" aria-hidden="true" />
                </span>
                <div className="space-y-1 pt-0.5">
                    <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
                    <p className="text-muted-foreground">{intro}</p>
                </div>
            </div>
            <ul className="space-y-3 rounded-xl bg-muted/70 p-4 text-sm">
                {points.map(point => (
                    <li key={point} className="flex gap-3">
                        <span className="mt-px grid size-5 shrink-0 place-items-center rounded-full bg-primary text-primary-foreground">
                            <Check className="size-3" strokeWidth={3} aria-hidden="true" />
                        </span>
                        <span className="leading-relaxed">{point}</span>
                    </li>
                ))}
            </ul>
            <div className="flex flex-col gap-2 sm:flex-row-reverse">
                <Button className="h-11 flex-1" onClick={onAccept}><Camera aria-hidden="true" />{actionLabel}</Button>
                <Button asChild variant="outline" className="h-11 sm:w-28"><Link href={backHref}>Back</Link></Button>
            </div>
        </div>
    );
}
