import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-6 px-4 py-8 text-center">
      <h1 className="text-3xl font-bold">Facial Recognition System</h1>
      <div className="max-w-xl space-y-2 text-muted-foreground">
        <p>
          Sign in with your face. Photos and video stay on your device. Only a numeric face signature is sent
          to the server.
        </p>
        <p className="text-sm">This is a demo, so there are no real accounts behind it.</p>
      </div>
      <div className="flex w-full max-w-sm flex-col gap-2">
        <Button asChild className="h-11"><Link href="/face/login">Sign in with your face</Link></Button>
        <Button asChild variant="outline" className="h-11"><Link href="/registration">Create account</Link></Button>
        <Button asChild variant="ghost" className="h-11"><Link href="/examples">Examples</Link></Button>
      </div>
    </main>
  );
}
