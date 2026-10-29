// Loads the golden values shared with the Python reference and the Flutter app (tests only).
import { readFileSync } from "node:fs";
import path from "node:path";

export interface TestVectors {
    alignment: { landmarks: number[][]; matrix: number[][] }[];
    pose: {
        points: Record<"left_eye" | "right_eye" | "forehead" | "chin", number[]>;
        pose: { yaw: number; pitch: number; roll: number };
        label: string | null;
    }[];
    classify: { pose: { yaw: number; pitch: number; roll: number }; label: string | null }[];
    preprocess: { width: number; height: number; rgb: number[]; tensor: number[] };
    embedding: {
        a: number[]; b: number[]; c: number[];
        a_normalized: number[];
        cosine_ab: number;
        fused_abc: number[];
        base64_of_1_2_3: string;
    };
}

export function loadTestVectors(): TestVectors {
    const file = path.resolve(__dirname, "../../../../../shared/test-vectors/face-pipeline.json");
    return JSON.parse(readFileSync(file, "utf8"));
}
