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
import { useEffect, useState } from "react"
import { Loader2 } from "lucide-react"

export function RegistrationFrom() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [fullname, setFullname] = useState("");
  const [password, setPassword] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);

  const handleRegisterFace = async () => {
    router.push(`/face/registration`);
  };

  const handleRegistration = async () => {
    setIsProcessing(true);
    const query = new URLSearchParams({ email, fullname });
    router.push(`/face/registration?${query}`)
    setIsProcessing(false);
  };

  useEffect(() => {
    router.prefetch(`/face/registration`);
  }, []);

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
          <Input id="email" type="email" placeholder="email@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)} />
        </div>
        <div className="grid gap-2">
          <Label htmlFor="fullname">Full Name</Label>
          <Input id="fullname" type="text" placeholder="Full Name"
            value={fullname}
            onChange={(e) => setFullname(e.target.value)}
          />
        </div>
        <div className="grid gap-2">
          <Label htmlFor="password">Password</Label>
          <Input id="password" type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)} />
        </div>
      </CardContent>
      <CardFooter>
        {/* <div className="grid"> */}
          <Button className="w-full" onClick={handleRegistration}>
            {isProcessing ? (
              <Loader2 className="animate-spin" />
            ) : ("Register")}
          </Button>
          {/* <Button className="w-full blue-700" onClick={handleRegisterFace}>Register Face</Button> */}
        {/* </div> */}
      </CardFooter>
    </Card>
  )
}