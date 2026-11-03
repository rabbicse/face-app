import type { Metadata } from "next"
import { RegistrationForm } from "@/components/forms/RegistrationForm"

export const metadata: Metadata = { title: "Create an account" }

export default function Page() {
  return (
    <main className="flex w-full flex-1 items-start justify-center px-4 pb-16 pt-4 sm:items-center sm:pt-8">
      <RegistrationForm />
    </main>
  )
}
