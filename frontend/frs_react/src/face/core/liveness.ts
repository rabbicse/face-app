export interface BlinkOptions {
    /** eyes count as open below this "closed" score (0 = wide open, 1 = shut) */
    openBelow: number;
    /** eyes count as closed above this score */
    closedAbove: number;
    /** a closure longer than this is not a blink (eyes shut on purpose / tracking glitch) */
    maxClosedMs: number;
}

export const DEFAULT_BLINK_OPTIONS: BlinkOptions = {
    openBelow: 0.3,
    closedAbove: 0.5,
    maxClosedMs: 700,
};

/**
 * Detects an open -> closed -> open eye cycle. A printed photo or a still image on a screen cannot
 * blink, so requiring one recent blink before identification stops the simplest presentation attack.
 * It is not a defence against replayed videos or a tampered client (see docs/client-side-embeddings.md).
 */
export class BlinkDetector {
    private state: "unknown" | "open" | "closed" = "unknown";
    private closedSince = 0;
    private lastBlink = -Infinity;
    private blinks = 0;

    constructor(private readonly options: BlinkOptions = DEFAULT_BLINK_OPTIONS) { }

    get count(): number {
        return this.blinks;
    }

    /**
     * @param closedScore how closed the eyes are, 0..1 (use the *less* closed eye, so a wink does not count)
     * @return true when this frame completes a blink
     */
    update(closedScore: number, timestampMs: number): boolean {
        const o = this.options;
        if (closedScore < o.openBelow) {
            const blinked = this.state === "closed" && timestampMs - this.closedSince <= o.maxClosedMs;
            this.state = "open";
            if (blinked) {
                this.lastBlink = timestampMs;
                this.blinks++;
            }
            return blinked;
        }
        if (closedScore > o.closedAbove && this.state === "open") {
            this.state = "closed";
            this.closedSince = timestampMs;
        }
        return false;
    }

    hasRecentBlink(timestampMs: number, windowMs: number): boolean {
        return timestampMs - this.lastBlink <= windowMs;
    }

    /** Call when the face is lost, so a blink of one face cannot unlock another. */
    reset(): void {
        this.state = "unknown";
        this.lastBlink = -Infinity;
    }
}
