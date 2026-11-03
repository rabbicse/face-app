import Link from "next/link";
import { ArrowRight, Binary, Camera, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { FacePreloader } from "@/face/react/FacePreloader";

const FACTS = [
  { icon: Camera, title: "Photos and video stay on your device", body: "Your browser finds and measures your face itself." },
  { icon: Binary, title: "Only a numeric face signature is sent", body: "The server stores numbers and compares them. It never receives an image." },
  { icon: Trash2, title: "You can delete your face data", body: "After you sign in, one button removes it from the server." },
];

export default function Home() {
  return (
    <main className="w-full flex-1 px-4 pb-10 sm:px-6">
      {/* most visitors go on to sign in: load the face models while they read, so the camera starts at once */}
      <FacePreloader />
      <section className="mx-auto grid w-full max-w-5xl items-center gap-10 pb-6 pt-6 md:grid-cols-[1.05fr_0.95fr] md:gap-14 md:pt-16 lg:pt-24">
        <div className="space-y-7">
          <div className="space-y-4">
            <h1 className="text-4xl font-semibold tracking-[-0.03em] sm:text-5xl">
              Sign in with your face.
              <span className="block text-primary">Keep your photos.</span>
            </h1>
            <p className="max-w-md text-lg leading-relaxed text-muted-foreground">
              Face recognition runs on your device. Only a numeric face signature is sent to the server.
            </p>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row">
            <Button asChild className="h-12 px-6 text-base">
              <Link href="/face/login">Sign in with your face <ArrowRight aria-hidden="true" /></Link>
            </Button>
            <Button asChild variant="outline" className="h-12 px-6 text-base">
              <Link href="/registration">Create account</Link>
            </Button>
          </div>
          <p className="text-sm text-muted-foreground">
            This is a demo, so there are no real accounts behind it.{" "}
            <Link href="/examples" className="link">See how it works</Link>
          </p>
        </div>

        <ul className="space-y-1 rounded-3xl bg-primary-soft p-3 sm:p-4" aria-label="What happens to your data">
          {FACTS.map(({ icon: Icon, title, body }) => (
            <li key={title} className="flex gap-4 rounded-2xl p-4">
              <span className="grid size-10 shrink-0 place-items-center rounded-xl bg-card text-primary shadow-[0_1px_2px_hsl(var(--ink)/0.08)]">
                <Icon className="size-5" aria-hidden="true" />
              </span>
              <div className="space-y-1">
                <p className="font-semibold text-primary-soft-foreground">{title}</p>
                <p className="text-sm leading-relaxed text-primary-soft-foreground/85">{body}</p>
              </div>
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
