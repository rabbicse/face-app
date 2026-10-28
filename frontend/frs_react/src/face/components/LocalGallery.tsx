"use client";

import { useEffect, useState } from "react";
import { LocalRepository } from "../repository";
import { Button } from "@/components/ui/button";

type Row = Awaited<ReturnType<LocalRepository["list"]>>[number];

/** Manage faces stored in this browser (NEXT_PUBLIC_FACE_MODE=local). */
export function LocalGallery() {
    const [repo] = useState(() => new LocalRepository());
    const [rows, setRows] = useState<Row[]>([]);
    const refresh = () => repo.list().then(setRows);
    useEffect(() => { void refresh(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

    const download = async () => {
        const blob = new Blob([await repo.exportJson()], { type: "application/json" });
        const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "face-gallery.json" });
        a.click();
    };

    return (
        <div className="w-full max-w-2xl space-y-4">
            <h1 className="text-2xl font-semibold tracking-tight">Faces stored in this browser</h1>
            <p className="text-sm text-muted-foreground">Used when the app runs with <code>NEXT_PUBLIC_FACE_MODE=local</code>. Only embeddings are stored (IndexedDB), no images.</p>
            <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={download}>Export JSON</Button>
                <label className="cursor-pointer rounded-md border px-3 py-1.5 text-sm">
                    Import JSON
                    <input hidden type="file" accept="application/json" onChange={async e => {
                        const f = e.target.files?.[0];
                        if (f) { await repo.importJson(await f.text()); await refresh(); }
                    }} />
                </label>
            </div>
            <ul className="divide-y rounded-xl border">
                {rows.length === 0 && <li className="p-4 text-sm text-muted-foreground">No faces registered yet.</li>}
                {rows.map(r => (
                    <li key={r.personId} className="flex items-center justify-between p-4">
                        <div>
                            <div className="font-medium">{String(r.person.name)}</div>
                            <div className="text-xs text-muted-foreground">{r.templates.map(t => t.label).join(" · ")} · {new Date(r.createdAt).toLocaleString()}</div>
                        </div>
                        <Button size="sm" variant="destructive" onClick={async () => { await repo.remove(r.personId); await refresh(); }}>Delete</Button>
                    </li>
                ))}
            </ul>
        </div>
    );
}
