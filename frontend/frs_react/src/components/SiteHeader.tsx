import Link from "next/link";
import { ScanFace } from "lucide-react";

/** Brand mark and a way home, on every page. */
export function SiteHeader() {
    return (
        <header className="w-full px-4 sm:px-6">
            <div className="mx-auto flex h-16 max-w-5xl items-center justify-between">
                <Link href="/" className="group flex items-center gap-2.5 rounded-md font-semibold tracking-tight no-underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2">
                    <span className="grid size-8 place-items-center rounded-lg bg-primary text-primary-foreground shadow-[0_1px_2px_hsl(var(--ink)/0.2)] transition-colors group-hover:bg-primary-hover">
                        <ScanFace className="size-[18px]" aria-hidden="true" strokeWidth={2.2} />
                    </span>
                    <span>Facial Recognition System</span>
                </Link>
                <Link href="/examples" className="hidden rounded-md px-3 py-2 text-sm font-medium text-muted-foreground no-underline transition-colors hover:bg-accent hover:text-accent-foreground sm:inline-block">
                    Examples
                </Link>
            </div>
        </header>
    );
}
