import { describe, expect, it } from "vitest";
import { isTfliteInfoMessage } from "./landmarker";

describe("isTfliteInfoMessage", () => {
    it("matches MediaPipe's TensorFlow Lite start-up notes", () => {
        expect(isTfliteInfoMessage(["INFO: Created TensorFlow Lite XNNPACK delegate for CPU."])).toBe(true);
        expect(isTfliteInfoMessage(["INFO: Initialized TensorFlow Lite runtime."])).toBe(true);
    });

    it("lets every other error through", () => {
        expect(isTfliteInfoMessage(["frame processing failed", new Error("x")])).toBe(false);
        expect(isTfliteInfoMessage(["ERROR: TensorFlow Lite failed"])).toBe(false);
        expect(isTfliteInfoMessage([new Error("INFO: Created TensorFlow Lite")])).toBe(false);
        expect(isTfliteInfoMessage([])).toBe(false);
    });
});
