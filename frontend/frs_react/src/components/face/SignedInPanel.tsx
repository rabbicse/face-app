"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { deletePerson } from "@/face/api";
import { clearSignedIn, readSignedIn } from "@/face/react/demoSession";
import { useHydrated } from "@/face/react/useHydrated";

type DeleteState = "idle" | "confirming" | "deleting" | "failed" | "deleted";

export function SignedInPanel() {
    const router = useRouter();
    const loaded = useHydrated();
    const person = useMemo(() => (loaded ? readSignedIn() : null), [loaded]);
    const [deleteState, setDeleteState] = useState<DeleteState>("idle");

    const leave = (href: string) => {
        clearSignedIn();
        router.push(href);
    };

    const deleteFaceData = async () => {
        if (!person) return;
        setDeleteState("deleting");
        try {
            await deletePerson(person.personId);
            clearSignedIn();
            setDeleteState("deleted");
        } catch (e) {
            console.warn("delete failed", e);
            setDeleteState("failed");
        }
    };

    if (!loaded) return null;

    if (deleteState === "deleted") {
        return (
            <Card className="w-full max-w-md">
                <CardContent className="space-y-4 p-6">
                    <div role="status" className="space-y-1">
                        <h1 className="text-xl font-semibold">Your face data is deleted</h1>
                        <p className="text-muted-foreground">
                            The numeric face signatures, your name and your email were removed from the server.
                        </p>
                    </div>
                    <Button asChild className="h-11 w-full"><Link href="/">Back to start</Link></Button>
                </CardContent>
            </Card>
        );
    }

    if (!person) {
        return (
            <Card className="w-full max-w-md">
                <CardContent className="space-y-4 p-6">
                    <h1 className="text-xl font-semibold">You&apos;re not signed in</h1>
                    <p className="text-muted-foreground">Sign in with your face to see this page.</p>
                    <Button asChild className="h-11 w-full"><Link href="/face/login">Sign in with your face</Link></Button>
                </CardContent>
            </Card>
        );
    }

    return (
        <Card className="w-full max-w-md">
            <CardContent className="space-y-6 p-6">
                <div className="space-y-1">
                    <h1 className="text-xl font-semibold break-words">Signed in as {person.name || person.email || "you"}</h1>
                    {person.email && <p className="break-words text-muted-foreground">{person.email}</p>}
                    {person.score !== null && (
                        <p className="text-sm text-muted-foreground">
                            Match score <span className="tabular-nums">{person.score.toFixed(2)}</span>
                        </p>
                    )}
                </div>

                <p className="text-sm text-muted-foreground">
                    This is a demo, so there is no account behind this page. It only remembers your match in this
                    browser tab.
                </p>

                <div className="flex flex-col gap-2">
                    <Button className="h-11" onClick={() => leave("/")}>Sign out</Button>
                    <Button variant="outline" className="h-11" onClick={() => leave("/face/login")}>Not you? Try again</Button>
                </div>

                <div className="space-y-3 border-t pt-4">
                    {deleteState === "idle" && (
                        <Button variant="outline" className="h-11 w-full text-red-700 hover:text-red-700"
                            onClick={() => setDeleteState("confirming")}>
                            Delete my face data
                        </Button>
                    )}
                    {deleteState !== "idle" && (
                        <div className="space-y-3">
                            <p className="text-sm" role={deleteState === "failed" ? "alert" : undefined}>
                                {deleteState === "failed"
                                    ? "Your face data could not be deleted. Check your connection, then try again."
                                    : "This removes your face signatures, name and email from the server. You would need to register again to sign in."}
                            </p>
                            <div className="flex flex-col gap-2 sm:flex-row">
                                <Button variant="destructive" className="h-11 flex-1" disabled={deleteState === "deleting"}
                                    onClick={deleteFaceData}>
                                    {deleteState === "deleting" ? "Deleting..." : "Delete permanently"}
                                </Button>
                                <Button variant="outline" className="h-11" disabled={deleteState === "deleting"}
                                    onClick={() => setDeleteState("idle")}>
                                    Keep my data
                                </Button>
                            </div>
                        </div>
                    )}
                </div>
            </CardContent>
        </Card>
    );
}
