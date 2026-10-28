import Link from "next/link";
import { ScanFace, UserPlus, Images, Gauge, Database } from "lucide-react";

const LINKS = [
    { href: "/registration", title: "Register", desc: "Create an account and enrol your face from 5 angles (yaw / pitch / roll guided).", icon: UserPlus },
    { href: "/face/login", title: "Face login", desc: "Identify yourself by just looking at the camera.", icon: ScanFace },
    { href: "/face/compare", title: "Compare photos", desc: "Example: verify two photos fully in the browser.", icon: Images },
    { href: "/face/benchmark", title: "Benchmark", desc: "Example: measure tracker / detector / embedding speed on this device.", icon: Gauge },
    { href: "/face/gallery", title: "Local gallery", desc: "Example: faces stored in this browser (local mode).", icon: Database },
];

export default function Home() {
    return (
        <main className="mx-auto max-w-3xl px-4 py-12">
            <h1 className="text-3xl font-semibold tracking-tight">Face Recognition</h1>
            <p className="mt-2 text-muted-foreground">
                Detection, alignment and 512-d ArcFace embeddings run on your device (WebGPU / WASM).
                Only vectors are sent to the server — never images.
            </p>
            <div className="mt-8 grid gap-3 sm:grid-cols-2">
                {LINKS.map(({ href, title, desc, icon: Icon }) => (
                    <Link key={href} href={href} className="group rounded-xl border p-5 transition hover:border-zinc-900">
                        <Icon className="h-6 w-6" />
                        <div className="mt-3 font-medium group-hover:underline">{title}</div>
                        <div className="text-sm text-muted-foreground">{desc}</div>
                    </Link>
                ))}
            </div>
            <p className="mt-6 text-xs text-muted-foreground">Add <code>?debug</code> to the login / registration URL to see live timings.</p>
        </main>
    );
}
