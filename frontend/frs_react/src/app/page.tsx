import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-6 px-4 text-center">
      <h1 className="text-3xl font-bold">Facial Recognition System</h1>
      <p className="max-w-xl text-muted-foreground">
        Faces are detected, aligned and turned into 512-d embeddings on your device. Only the embedding is
        sent to the server, never a photo.
      </p>
      <div className="flex flex-wrap justify-center gap-3">
        <Button asChild><Link href="/login">Login</Link></Button>
        <Button asChild variant="outline"><Link href="/registration">Create account</Link></Button>
        <Button asChild variant="outline"><Link href="/face/login">Login with face</Link></Button>
        <Button asChild variant="ghost"><Link href="/examples">Examples</Link></Button>
      </div>
    </main>
  );
}
