/** Tiny external store for useSyncExternalStore: per-frame code writes, React reads. */
export interface Store<S> {
    get(): S;
    set(patch: Partial<S>): void;
    subscribe(fn: () => void): () => void;
}

export function createStore<S extends object>(initial: S): Store<S> {
    let state = initial;
    const listeners = new Set<() => void>();
    return {
        get: () => state,
        set(patch) {
            let changed = false;
            for (const k in patch) {
                if (!Object.is(state[k as keyof S], patch[k as keyof S])) { changed = true; break; }
            }
            if (!changed) return;          // no re-render when nothing changed
            state = { ...state, ...patch };
            listeners.forEach(l => l());
        },
        subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    };
}
