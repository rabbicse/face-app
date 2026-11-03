import Link from "next/link";
import { ArrowUpRight, Images, Activity, Zap } from "lucide-react";

const EXAMPLES = [
    {
        href: "/examples/pipeline",
        icon: Activity,
        title: "Live pipeline inspector",
        description: "Webcam -> landmarks -> head pose -> aligned crop -> 512-d embedding, with timings, quality metrics and live similarity to a reference capture.",
    },
    {
        href: "/examples/compare",
        icon: Images,
        title: "Compare two photos",
        description: "Pick two images; both embeddings are computed in your browser and compared with cosine similarity. Nothing is uploaded.",
    },
    {
        href: "/face/registration?plan=quick&fullname=Demo&email=demo@example.com",
        icon: Zap,
        title: "Quick registration",
        description: "Frontal only enrolment (3 samples) - handy for kiosks or quick tests. Samples must differ slightly, so a frozen image cannot finish it.",
    },
];

export default function ExamplesPage() {
    return (
        <main className="mx-auto w-full max-w-3xl flex-1 space-y-8 px-4 pb-16 pt-6 sm:px-6 sm:pt-10">
            <div className="space-y-2">
                <h1 className="text-3xl font-semibold tracking-tight">On-device face recognition examples</h1>
                <p className="max-w-xl text-muted-foreground">Everything on these pages runs in your browser.</p>
            </div>
            <ul className="divide-y overflow-hidden rounded-2xl border bg-card shadow-card">
                {EXAMPLES.map(({ href, title, description, icon: Icon }) => (
                    <li key={href}>
                        <Link href={href} className="group flex gap-4 p-5 no-underline transition-colors hover:bg-accent focus-visible:bg-accent focus-visible:outline-none sm:p-6">
                            <span className="grid size-11 shrink-0 place-items-center rounded-xl bg-primary-soft text-primary transition-colors group-hover:bg-primary group-hover:text-primary-foreground">
                                <Icon className="size-5" aria-hidden="true" />
                            </span>
                            <span className="min-w-0 flex-1 space-y-1">
                                <span className="flex items-center gap-1.5 font-semibold text-foreground group-hover:text-primary">
                                    {title}
                                    <ArrowUpRight className="size-4 opacity-0 transition-opacity group-hover:opacity-100" aria-hidden="true" />
                                </span>
                                <span className="block text-sm leading-relaxed text-muted-foreground">{description}</span>
                            </span>
                        </Link>
                    </li>
                ))}
            </ul>
            <Link href="/" className="link inline-block text-sm">Back to start</Link>
        </main>
    );
}
