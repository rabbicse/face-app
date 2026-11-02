import type { Metadata } from "next"
import { RegistrationForm } from "@/components/forms/RegistrationForm"

export const metadata: Metadata = { title: "Create an account" }

export default function Page() {
  return (
    <main className="flex min-h-screen w-full items-center justify-center px-4 py-8">
      <RegistrationForm />
    </main>
  )
}
