"use client";

import { useSyncExternalStore } from "react";

const subscribe = () => () => {};

/** False during server rendering and hydration, then true: the point where sessionStorage can be read. */
export function useHydrated(): boolean {
    return useSyncExternalStore(subscribe, () => true, () => false);
}
