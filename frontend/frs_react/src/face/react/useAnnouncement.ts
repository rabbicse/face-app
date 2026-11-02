"use client";

import { useEffect, useState } from "react";

/**
 * Frame-driven guidance can change several times a second. Screen readers only get a message once it has
 * stayed the same for `settleMs`, so they read steady instructions instead of interrupting themselves.
 */
export function useAnnouncement(message: string, settleMs = 1200): string {
    const [announced, setAnnounced] = useState("");

    useEffect(() => {
        const timer = setTimeout(() => setAnnounced(message), settleMs);
        return () => clearTimeout(timer);
    }, [message, settleMs]);

    return announced;
}
