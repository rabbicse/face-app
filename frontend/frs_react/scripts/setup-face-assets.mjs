#!/usr/bin/env node
/**
 * Puts the on-device face recognition assets into public/ (skips files that already exist):
 *   public/ort/                         ONNX Runtime Web wasm binaries (from node_modules)
 *   public/mediapipe/wasm/              MediaPipe Tasks Vision wasm (from node_modules)
 *   public/models/mediapipe/face_landmarker.task
 *   public/models/arcface/w600k_mbf.onnx  (MobileFaceNet ArcFace, 512-d, from InsightFace buffalo_s)
 *   public/models/arcface/w600k_mbf.fp16.onnx  same model with float16 weight storage (6.9 MB instead of
 *       13.6 MB, identical results) - the file the browser downloads
 *
 * The InsightFace pretrained models are released for non-commercial research use only. To use your
 * own weights export them to ONNX (see the stable-v01 branch for the PyTorch exporter) and set NEXT_PUBLIC_EMBEDDING_MODEL_URL.
 *
 * Usage: node scripts/setup-face-assets.mjs [--force]
 *
 * On deploy builds (VERCEL=1 or FACE_ASSETS_DEPLOY=1) only what the deployed site loads is put into public/: the fp32 model stays out of
 * the deployment and the WebGPU runtime is skipped unless NEXT_PUBLIC_EMBEDDING_WEBGPU=true.
 */
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { inflateRawSync } from "node:zlib";
import { convertWeightsToFp16 } from "./onnx-fp16.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const pub = path.join(root, "public");
const force = process.argv.includes("--force");
// slim asset set for deployments: Vercel builds and the Docker image (FACE_ASSETS_DEPLOY=1)
const deployBuild = process.env.VERCEL === "1" || process.env.FACE_ASSETS_DEPLOY === "1";
const webgpu = process.env.NEXT_PUBLIC_EMBEDDING_WEBGPU === "true";

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
        f => /^ort-wasm.*\.(wasm|mjs)$/.test(f) && (!deployBuild || webgpu || !f.includes(".jsep.")));
    copyDir(path.join(root, "node_modules/@mediapipe/tasks-vision/wasm"), path.join(pub, "mediapipe/wasm"));

    const landmarker = path.join(pub, "models/mediapipe/face_landmarker.task");
    if (force || !existsSync(landmarker)) {
        mkdirSync(path.dirname(landmarker), { recursive: true });
        writeFileSync(landmarker, await download(LANDMARKER_URL));
    }

    // the browser loads the fp16 copy; on deploy builds the fp32 original is kept out of public/
    const arcface = deployBuild
        ? path.join(os.tmpdir(), "face-assets", "w600k_mbf.onnx")
        : path.join(pub, "models/arcface/w600k_mbf.onnx");
    if (force || !existsSync(arcface)) {
        mkdirSync(path.dirname(arcface), { recursive: true });
        const onnx = extractFromZip(await download(BUFFALO_S_URL), "w600k_mbf.onnx");
        const digest = createHash("sha256").update(onnx).digest("hex");
        if (digest !== W600K_MBF_SHA256) throw new Error(`w600k_mbf.onnx checksum mismatch: ${digest}`);
        writeFileSync(arcface, onnx);
    } else if (createHash("sha256").update(readFileSync(arcface)).digest("hex") !== W600K_MBF_SHA256) {
        console.warn("  note: public/models/arcface/w600k_mbf.onnx differs from the InsightFace release (custom model?)");
    }
    const fp16 = path.join(pub, "models/arcface/w600k_mbf.fp16.onnx");
    if (force || !existsSync(fp16)) {
        mkdirSync(path.dirname(fp16), { recursive: true });
        const { model, converted } = convertWeightsToFp16(new Uint8Array(readFileSync(arcface)));
        writeFileSync(fp16, model);
        console.log(`  ${path.relative(root, fp16)}: ${converted} weight tensors stored as float16`);
    }
    console.log("Done.");
}

main().catch(e => {
    console.error(`Face asset setup failed: ${e.message}`);
    process.exit(1);
});
