import { describe, expect, it } from "vitest";
import { describeCameraError } from "./cameraErrors";

const named = (name: string) => Object.assign(new Error("raw browser text"), { name });

describe("describeCameraError", () => {
    it.each([
        ["NotAllowedError", /blocked/],
        ["SecurityError", /blocked/],
        ["NotFoundError", /No camera/],
        ["OverconstrainedError", /No camera/],
        ["NotReadableError", /another app/],
        ["AbortError", /another app/],
        ["InsecureContextError", /https/],
    ])("explains %s", (name, expected) => {
        expect(describeCameraError(named(name))).toMatch(expected);
    });

    it("never shows the raw browser message", () => {
        expect(describeCameraError(named("SomethingNew"))).not.toContain("raw browser text");
        expect(describeCameraError("not an error")).toMatch(/could not start/);
    });
});
