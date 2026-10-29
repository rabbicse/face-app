#!/usr/bin/env node
/**
 * Puts the on-device face recognition assets into public/ (skips files that already exist):
 *   public/ort/                         ONNX Runtime Web wasm binaries (from node_modules)
 *   public/mediapipe/wasm/              MediaPipe Tasks Vision wasm (from node_modules)
 *   public/models/mediapipe/face_landmarker.task
 *   public/models/arcface/w600k_mbf.onnx  (MobileFaceNet ArcFace, 512-d, from InsightFace buffalo_s)
 *
 * The InsightFace pretrained models are released for non-commercial research use only. To use your
 * own weights export them with backend/frs/tools/export_onnx.py and set NEXT_PUBLIC_EMBEDDING_MODEL_URL.
 *
 * Usage: node scripts/setup-face-assets.mjs [--force]
 */
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { inflateRawSync } from "node:zlib";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const pub = path.join(root, "public");
const force = process.argv.includes("--force");

const LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const BUFFALO_S_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip";
const W600K_MBF_SHA256 = "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f";

function copyDir(from, to, filter = () => true) {
    mkdirSync(to, { recursive: true });
    let n = 0;
    for (const f of readdirSync(from)) {
        if (!filter(f)) continue;
        const dst = path.join(to, f);
        if (force || !existsSync(dst)) {
            copyFileSync(path.join(from, f), dst);
            n++;
        }
    }
    console.log(`  ${path.relative(root, to)}: ${n} file(s) copied`);
}

async function download(url) {
    console.log(`  downloading ${url}`);
    const res = await fetch(url, { redirect: "follow" });
    if (!res.ok) throw new Error(`GET ${url} -> ${res.status}`);
    return Buffer.from(await res.arrayBuffer());
}

/** Minimal zip reader (stored / deflate entries) - avoids an extra dependency. */
function extractFromZip(zip, wantedName) {
    let eocd = zip.length - 22;
    while (eocd >= 0 && zip.readUInt32LE(eocd) !== 0x06054b50) eocd--;
    if (eocd < 0) throw new Error("not a zip file");
    const count = zip.readUInt16LE(eocd + 10);
    let p = zip.readUInt32LE(eocd + 16);
    for (let i = 0; i < count; i++) {
        const method = zip.readUInt16LE(p + 10);
        const compSize = zip.readUInt32LE(p + 20);
        const nameLen = zip.readUInt16LE(p + 28);
        const extraLen = zip.readUInt16LE(p + 30);
        const commentLen = zip.readUInt16LE(p + 32);
        const localOffset = zip.readUInt32LE(p + 42);
        const name = zip.toString("utf8", p + 46, p + 46 + nameLen);
        if (path.basename(name) === wantedName) {
            const lNameLen = zip.readUInt16LE(localOffset + 26);
            const lExtraLen = zip.readUInt16LE(localOffset + 28);
            const start = localOffset + 30 + lNameLen + lExtraLen;
            const data = zip.subarray(start, start + compSize);
            return method === 0 ? data : inflateRawSync(data);
        }
        p += 46 + nameLen + extraLen + commentLen;
    }
    throw new Error(`${wantedName} not found in archive`);
}

async function main() {
    console.log("Setting up face recognition assets");
    copyDir(path.join(root, "node_modules/onnxruntime-web/dist"), path.join(pub, "ort"),
        f => /^ort-wasm.*\.(wasm|mjs)$/.test(f));
    copyDir(path.join(root, "node_modules/@mediapipe/tasks-vision/wasm"), path.join(pub, "mediapipe/wasm"));

    const landmarker = path.join(pub, "models/mediapipe/face_landmarker.task");
    if (force || !existsSync(landmarker)) {
        mkdirSync(path.dirname(landmarker), { recursive: true });
        writeFileSync(landmarker, await download(LANDMARKER_URL));
    }

    const arcface = path.join(pub, "models/arcface/w600k_mbf.onnx");
    if (force || !existsSync(arcface)) {
        mkdirSync(path.dirname(arcface), { recursive: true });
        const onnx = extractFromZip(await download(BUFFALO_S_URL), "w600k_mbf.onnx");
        const digest = createHash("sha256").update(onnx).digest("hex");
        if (digest !== W600K_MBF_SHA256) throw new Error(`w600k_mbf.onnx checksum mismatch: ${digest}`);
        writeFileSync(arcface, onnx);
    } else if (createHash("sha256").update(readFileSync(arcface)).digest("hex") !== W600K_MBF_SHA256) {
        console.warn("  note: public/models/arcface/w600k_mbf.onnx differs from the InsightFace release (custom model?)");
    }
    console.log("Done.");
}

main().catch(e => {
    console.error(`Face asset setup failed: ${e.message}`);
    process.exit(1);
});
