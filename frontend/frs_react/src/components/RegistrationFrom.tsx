"use client"

import { Icons } from "@/components/ui/icons"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useRouter } from "next/navigation"

export function RegistrationFrom() {
  const router = useRouter();
  const handleRegisterFace = async () => {
    router.push(`/face/registration`);
  };

  return (
    <Card>
      <CardHeader className="space-y-1">
        <CardTitle className="text-2xl">Create an account</CardTitle>
        <CardDescription>
          Enter your account information below to registration
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div className="grid gap-2">
          <Label htmlFor="email">Email</Label>
          <Input id="email" type="email" placeholder="email@example.com" />
        </div>
        <div className="grid gap-2">
          <Label htmlFor="fullname">Full Name</Label>
          <Input id="fullname" type="text" placeholder="Full Name" />
        </div>
        <div className="grid gap-2">
          <Label htmlFor="password">Password</Label>
          <Input id="password" type="password" />
        </div>
      </CardContent>
      <CardFooter>
        <div className="grid grid-cols-2 gap-6">
          <Button className="w-full">Create account</Button>
          <Button className="w-full" onClick={handleRegisterFace}>Register Face</Button>
        </div>
      </CardFooter>
    </Card>
  )
}