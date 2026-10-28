/**
 * Where enrolled faces live. Two interchangeable implementations:
 *   ServerRepository – FastAPI /api/v2 (Qdrant). Only 512-float vectors leave the browser.
 *   LocalRepository  – IndexedDB in this browser; nothing leaves the device at all.
 */
import { FACE_CONFIG } from "./runtime/config";
import { Candidate, decide, Decision, DEFAULT_DECISION, Gallery } from "./core/identification";
import { EMBEDDING_MODEL_ID, embeddingToBase64 } from "./core/vector";
import type { EnrollmentSample } from "./core/enrollment";

export interface PersonInfo {
    name: string;
    email?: string;
    phone?: string;
    [k: string]: unknown;
}

export interface Enrolled { personId: string; templates: number }

export interface IdentifyResponse {
    decision: Decision<PersonInfo>;
    candidates: Candidate<PersonInfo>[];
}

export interface FaceRepository {
    readonly kind: "server" | "local";
    register(person: PersonInfo, templates: EnrollmentSample[]): Promise<Enrolled>;
    identify(embedding: Float32Array): Promise<IdentifyResponse>;
}

// --------------------------------------------------------------------------------------
export class ServerRepository implements FaceRepository {
    readonly kind = "server" as const;
    constructor(private baseUrl = FACE_CONFIG.apiUrl) { }

    private async post<T>(path: string, body: unknown): Promise<T> {
        let res: Response;
        try {
            res = await fetch(`${this.baseUrl}${path}`, {
                method: "POST",
                headers: { "content-type": "application/json", accept: "application/json" },
                body: JSON.stringify(body),
            });
        } catch {
            throw new Error(`Cannot reach the face API at ${this.baseUrl}. Is the backend running? (or set NEXT_PUBLIC_FACE_MODE=local)`);
        }
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data?.detail?.toString?.() ?? `HTTP ${res.status}`);
        return data as T;
    }

    async register(person: PersonInfo, templates: EnrollmentSample[]): Promise<Enrolled> {
        const r = await this.post<{ person_id: string; templates: number }>("/api/v2/persons", {
            model: EMBEDDING_MODEL_ID,
            person,
            templates: templates.map(t => ({
                label: t.step,
                embedding: embeddingToBase64(t.embedding),
                pose: t.pose,
                quality: { sharpness: t.quality.sharpness, brightness: t.quality.brightness, inter_ocular_px: t.quality.interOcularPx },
            })),
        });
        return { personId: r.person_id, templates: r.templates };
    }

    async identify(embedding: Float32Array): Promise<IdentifyResponse> {
        type Row = { person_id: string; score: number; person: PersonInfo };
        const r = await this.post<{ status: "match" | "no_match"; reason?: string; candidates: Row[] }>("/api/v2/identify", {
            model: EMBEDDING_MODEL_ID,
            embedding: embeddingToBase64(embedding),
            top_k: 5,
        });
        const candidates = r.candidates.map(c => ({ personId: c.person_id, score: c.score, meta: c.person }));
        // server already decided; mirror it client-side so both modes return the same shape
        const decision: Decision<PersonInfo> = r.status === "match"
            ? { status: "match", candidate: candidates[0], runnerUp: candidates[1] }
            : { status: "no_match", best: candidates[0], reason: (r.reason as "below_threshold" | "ambiguous" | "empty_gallery") ?? "below_threshold" };
        return { decision, candidates };
    }
}

// --------------------------------------------------------------------------------------
const DB_NAME = "face-app", STORE = "persons";

interface StoredPerson { personId: string; person: PersonInfo; model: string; templates: { label: string; embedding: number[] }[]; createdAt: number }

function openDb(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, 1);
        req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: "personId" });
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

function tx<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
    return openDb().then(db => new Promise<T>((resolve, reject) => {
        const r = fn(db.transaction(STORE, mode).objectStore(STORE));
        r.onsuccess = () => resolve(r.result);
        r.onerror = () => reject(r.error);
    }));
}

export class LocalRepository implements FaceRepository {
    readonly kind = "local" as const;
    private gallery: Gallery<PersonInfo> | null = null;

    async list(): Promise<StoredPerson[]> {
        return (await tx<StoredPerson[]>("readonly", s => s.getAll())).filter(p => p.model === EMBEDDING_MODEL_ID);
    }

    async remove(personId: string): Promise<void> {
        await tx("readwrite", s => s.delete(personId));
        this.gallery = null;
    }

    private async loadGallery(): Promise<Gallery<PersonInfo>> {
        if (this.gallery) return this.gallery;
        const g = new Gallery<PersonInfo>();
        for (const p of await this.list()) {
            p.templates.forEach(t => g.add({ personId: p.personId, embedding: Float32Array.from(t.embedding), label: t.label, meta: p.person }));
        }
        return (this.gallery = g);
    }

    async register(person: PersonInfo, templates: EnrollmentSample[]): Promise<Enrolled> {
        const personId = crypto.randomUUID();
        const rec: StoredPerson = {
            personId, person, model: EMBEDDING_MODEL_ID, createdAt: Date.now(),
            templates: templates.map(t => ({ label: t.step, embedding: Array.from(t.embedding) })),
        };
        await tx("readwrite", s => s.put(rec));
        this.gallery = null;
        return { personId, templates: templates.length };
    }

    async identify(embedding: Float32Array): Promise<IdentifyResponse> {
        const candidates = (await this.loadGallery()).search(embedding, 5);
        return { decision: decide(candidates, DEFAULT_DECISION), candidates };
    }

    async exportJson(): Promise<string> { return JSON.stringify(await this.list()); }

    async importJson(json: string): Promise<number> {
        const rows = JSON.parse(json) as StoredPerson[];
        for (const r of rows) await tx("readwrite", s => s.put(r));
        this.gallery = null;
        return rows.length;
    }
}

/**
 * Storage mode: NEXT_PUBLIC_FACE_MODE at build time, overridable at runtime with
 * ?mode=local / ?mode=server (remembered in localStorage) to try the app without a backend.
 */
export function currentMode(): "server" | "local" {
    if (typeof window === "undefined") return FACE_CONFIG.mode;
    const q = new URLSearchParams(window.location.search).get("mode");
    try {
        if (q === "local" || q === "server") localStorage.setItem("face.mode", q);
        const saved = localStorage.getItem("face.mode");
        if (saved === "local" || saved === "server") return saved;
    } catch { /* storage blocked */ }
    return FACE_CONFIG.mode;
}

let repo: FaceRepository | null = null;
export function getRepository(): FaceRepository {
    const mode = currentMode();
    if (!repo || repo.kind !== mode) repo = mode === "local" ? new LocalRepository() : new ServerRepository();
    return repo;
}
