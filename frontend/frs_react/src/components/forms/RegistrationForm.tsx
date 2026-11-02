"use client"

import { FormEvent, useEffect, useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
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
    <Card className="w-full max-w-sm">
      <FacePreloader />
      <form onSubmit={handleSubmit}>
        <CardHeader className="space-y-1">
          <h1 className="text-2xl font-semibold leading-none tracking-tight">Create an account</h1>
          <CardDescription>
            Your name is used to greet you when you sign in. Your email identifies your face profile.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
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
        <CardFooter className="flex-col gap-4">
          <Button type="submit" className="h-11 w-full">Continue</Button>
          <p className="text-sm text-muted-foreground">
            Already registered?{" "}
            <Link href="/face/login" className="text-foreground underline underline-offset-4">Sign in with your face</Link>
          </p>
        </CardFooter>
      </form>
    </Card>
  )
}
