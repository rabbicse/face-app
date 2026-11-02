import { Metadata } from "next"
import { SignedInPanel } from "@/components/face/SignedInPanel"

export const metadata: Metadata = {
  title: "Signed in",
  description: "Who you were matched as, with options to sign out or delete your face data.",
}

export default function DashboardPage() {
  return (
    <main className="flex min-h-screen w-full items-center justify-center px-4 py-8">
      <SignedInPanel />
    </main>
  )
}
