import { redirect } from "next/navigation"

/** Face login is the only sign-in method in this demo. */
export default function Page() {
  redirect("/face/login")
}
