import { describe, expect, it } from "vitest";
import { BlinkDetector } from "./liveness";

/** feeds (score, t) pairs, returns the frames that completed a blink */
function run(d: BlinkDetector, frames: [number, number][]): number[] {
    return frames.flatMap(([score, t]) => (d.update(score, t) ? [t] : []));
}

describe("BlinkDetector", () => {
    it("detects open -> closed -> open", () => {
        const d = new BlinkDetector();
        expect(run(d, [[0.1, 0], [0.1, 33], [0.8, 66], [0.9, 100], [0.1, 133], [0.05, 166]])).toEqual([133]);
        expect(d.count).toBe(1);
        expect(d.hasRecentBlink(1000, 2000)).toBe(true);
        expect(d.hasRecentBlink(5000, 2000)).toBe(false);
    });

    it("needs to see the eyes open first (a face appearing with closed eyes is not a blink)", () => {
        const d = new BlinkDetector();
        expect(run(d, [[0.9, 0], [0.9, 33], [0.1, 66]])).toEqual([]);
    });

    it("ignores long closures, half closed eyes and a still photo", () => {
        const long = new BlinkDetector();
        expect(run(long, [[0.1, 0], [0.9, 100], [0.9, 1000], [0.1, 1100]])).toEqual([]);
        const squint = new BlinkDetector();
        expect(run(squint, [[0.1, 0], [0.45, 100], [0.1, 200]])).toEqual([]);
        const photo = new BlinkDetector();
        expect(run(photo, Array.from({ length: 100 }, (_, i) => [0.12, i * 33] as [number, number]))).toEqual([]);
    });

    it("forgets blinks when the face is lost", () => {
        const d = new BlinkDetector();
        run(d, [[0.1, 0], [0.9, 50], [0.1, 100]]);
        d.reset();
        expect(d.hasRecentBlink(200, 5000)).toBe(false);
    });
});
