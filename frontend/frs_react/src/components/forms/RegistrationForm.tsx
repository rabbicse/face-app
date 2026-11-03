"use client"

import { FormEvent, useEffect, useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { ArrowRight, UserRoundPlus } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { savePendingEnrolment } from "@/face/react/demoSession"
import { FacePreloader } from "@/face/react/FacePreloader"

export function RegistrationForm() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [fullname, setFullname] = useState("");

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    savePendingEnrolment({ fullname: fullname.trim(), email: email.trim() });
    router.push("/face/registration");
  };

  useEffect(() => {
    router.prefetch("/face/registration");
  }, [router]);

  return (
    <Card className="w-full max-w-md">
      <FacePreloader />
      <form onSubmit={handleSubmit}>
        <CardHeader className="space-y-3 p-6 sm:p-8 sm:pb-6">
          <span className="grid size-12 place-items-center rounded-2xl bg-primary-soft text-primary">
            <UserRoundPlus className="size-6" aria-hidden="true" />
          </span>
          <h1 className="text-2xl font-semibold tracking-tight">Create an account</h1>
          <CardDescription className="text-base leading-relaxed">
            Your name is used to greet you when you sign in. Your email identifies your face profile.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-5 px-6 sm:px-8">
          <div className="grid gap-2">
            <Label htmlFor="fullname">Full name</Label>
            <Input id="fullname" type="text" autoComplete="name" required
              value={fullname}
              onChange={(e) => setFullname(e.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="email">Email</Label>
            <Input id="email" type="email" autoComplete="email" placeholder="email@example.com" required
              value={email}
              onChange={(e) => setEmail(e.target.value)} />
          </div>
        </CardContent>
        <CardFooter className="flex-col gap-5 px-6 pb-6 sm:px-8 sm:pb-8">
          <Button type="submit" className="h-11 w-full">Continue to face scan <ArrowRight aria-hidden="true" /></Button>
          <p className="text-sm text-muted-foreground">
            Already registered?{" "}
            <Link href="/face/login" className="link">Sign in with your face</Link>
          </p>
        </CardFooter>
      </form>
    </Card>
  )
}
