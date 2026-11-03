import { Metadata } from "next"
import { SignedInPanel } from "@/components/face/SignedInPanel"

export const metadata: Metadata = {
  title: "Signed in",
  description: "Who you were matched as, with options to sign out or delete your face data.",
}

export default function DashboardPage() {
  return (
    <main className="flex w-full flex-1 items-start justify-center px-4 pb-16 pt-4 sm:items-center sm:pt-8">
      <SignedInPanel />
    </main>
  )
}
