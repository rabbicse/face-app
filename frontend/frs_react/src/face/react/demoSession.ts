/**
 * Demo only: there are no accounts or server sessions, so the match and the details typed before enrolment
 * live in sessionStorage. That keeps names and e-mails out of URLs, and the data ends with the browser tab.
 */
export interface SignedInPerson {
    personId: number;
    name: string;
    email?: string;
    score: number | null;
}

export interface PendingEnrolment {
    fullname: string;
    email: string;
}

const SIGNED_IN_KEY = "frs.signedIn";
const PENDING_KEY = "frs.pendingEnrolment";

function read<T>(key: string): T | null {
    try {
        const raw = sessionStorage.getItem(key);
        return raw ? (JSON.parse(raw) as T) : null;
    } catch {
        return null;
    }
}

function write(key: string, value: unknown): void {
    try {
        sessionStorage.setItem(key, JSON.stringify(value));
    } catch { /* blocked storage: the pages fall back to their empty states */ }
}

function remove(key: string): void {
    try {
        sessionStorage.removeItem(key);
    } catch { /* nothing stored */ }
}

export const readSignedIn = () => read<SignedInPerson>(SIGNED_IN_KEY);
export const saveSignedIn = (person: SignedInPerson) => write(SIGNED_IN_KEY, person);
export const clearSignedIn = () => remove(SIGNED_IN_KEY);

export const readPendingEnrolment = () => read<PendingEnrolment>(PENDING_KEY);
export const savePendingEnrolment = (pending: PendingEnrolment) => write(PENDING_KEY, pending);
export const clearPendingEnrolment = () => remove(PENDING_KEY);
