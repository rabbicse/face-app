"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { BadgeCheck, LogOut, RotateCcw, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { deletePerson } from "@/face/api";
import { clearSignedIn, readSignedIn, SignedInPerson } from "@/face/react/demoSession";
import { useHydrated } from "@/face/react/useHydrated";

/** Everything the server returned for this person, in reading order; missing details are left out. */
function detailRows(p: SignedInPerson): [string, string][] {
    const rows: [string, string | number | undefined | null][] = [
        ["Name", p.name],
        ["Email", p.email],
        ["Phone", p.phone],
        ["Age", p.age],
        ["Address", p.address],
        ["City", p.city],
        ["Country", p.country],
        ["Profile ID", p.personId],
        ["Match score", p.score === null ? null : p.score.toFixed(2)],
        ["Signed in", p.signedInAt ? new Date(p.signedInAt).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : null],
    ];
    return rows.filter((r): r is [string, string | number] => r[1] !== undefined && r[1] !== null && r[1] !== "")
        .map(([k, v]) => [k, String(v)]);
}

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
                <CardContent className="space-y-4 p-6 sm:p-8">
                    <div role="status" className="space-y-1">
                        <h1 className="text-2xl font-semibold tracking-tight">Your face data is deleted</h1>
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
                <CardContent className="space-y-4 p-6 sm:p-8">
                    <h1 className="text-2xl font-semibold tracking-tight">You&apos;re not signed in</h1>
                    <p className="text-muted-foreground">Sign in with your face to see this page.</p>
                    <Button asChild className="h-11 w-full"><Link href="/face/login">Sign in with your face</Link></Button>
                </CardContent>
            </Card>
        );
    }

    return (
        <Card className="w-full max-w-md overflow-hidden">
            <div className="h-20 bg-primary-soft" aria-hidden="true" />
            <CardContent className="-mt-10 space-y-6 p-6 sm:px-8">
                <div className="space-y-3">
                    <span aria-hidden="true"
                        className="grid size-20 place-items-center rounded-full border-4 border-card bg-primary text-3xl font-semibold text-primary-foreground shadow-card">
                        {(person.name || person.email || "?").trim().charAt(0).toUpperCase()}
                    </span>
                    <div className="space-y-1">
                        <h1 className="text-2xl font-semibold tracking-tight break-words">Signed in as {person.name || person.email || "you"}</h1>
                    </div>
                    {person.score !== null && (
                        <p className="inline-flex items-center gap-1.5 rounded-full bg-success-soft px-3 py-1 text-sm font-medium text-success">
                            <BadgeCheck className="size-4" aria-hidden="true" />
                            Face matched
                        </p>
                    )}
                </div>

                <section aria-labelledby="details-heading" className="space-y-2">
                    <h2 id="details-heading" className="text-sm font-semibold text-muted-foreground">Your details</h2>
                    <dl className="divide-y rounded-xl border text-sm">
                        {detailRows(person).map(([label, value]) => (
                            <div key={label} className="grid grid-cols-[7.5rem_1fr] gap-3 px-4 py-2.5">
                                <dt className="text-muted-foreground">{label}</dt>
                                <dd className="min-w-0 break-words font-medium tabular-nums">{value}</dd>
                            </div>
                        ))}
                    </dl>
                </section>

                <p className="text-sm text-muted-foreground">
                    This is a demo, so there is no account behind this page. It only remembers your match in this
                    browser tab.
                </p>

                <div className="flex flex-col gap-2">
                    <Button className="h-11" onClick={() => leave("/")}><LogOut aria-hidden="true" />Sign out</Button>
                    <Button variant="outline" className="h-11" onClick={() => leave("/face/login")}><RotateCcw aria-hidden="true" />Not you? Try again</Button>
                </div>

                <div className="space-y-3 border-t pt-4">
                    {deleteState === "idle" && (
                        <Button variant="ghost" className="h-11 w-full text-destructive hover:bg-destructive-soft hover:text-destructive"
                            onClick={() => setDeleteState("confirming")}>
                            <Trash2 aria-hidden="true" />Delete my face data
                        </Button>
                    )}
                    {deleteState !== "idle" && (
                        <div className="space-y-3 rounded-xl bg-destructive-soft p-4">
                            <p className={`text-sm ${deleteState === "failed" ? "font-medium text-destructive" : ""}`} role={deleteState === "failed" ? "alert" : undefined}>
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
