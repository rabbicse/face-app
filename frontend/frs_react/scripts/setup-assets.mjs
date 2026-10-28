#!/usr/bin/env node
/**
 * Prepares everything the browser needs to run face recognition on-device:
 *
 *   public/ort/         onnxruntime-web WASM/WebGPU binaries      (copied from node_modules)
 *   public/mediapipe/   MediaPipe Tasks Vision WASM               (copied from node_modules)
 *   public/models/      face_landmarker.task, det_500m.onnx, w600k_mbf.onnx  (downloaded)
 *
 * Usage:
 *   npm run setup:assets            # runtimes + models
 *   node scripts/setup-assets.mjs --runtime-only   (runs automatically on npm install)
 *
 * Self-hosting (instead of CDNs) keeps the app working offline and lets us enable
 * cross-origin isolation (multi-threaded WASM) without third-party CORP headers.
 *
 * NOTE: the InsightFace models (buffalo_s) are released for NON-COMMERCIAL research use.
 */
import { createWriteStream, existsSync, mkdirSync, copyFileSync, readdirSync, readFileSync, writeFileSync, statSync, renameSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { inflateRawSync } from "node:zlib";
import { createHash } from "node:crypto";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const PUBLIC = join(ROOT, "public");
const REPO_MODELS = resolve(ROOT, "..", "..", "models"); // shared with backend / examples
const runtimeOnly = process.argv.includes("--runtime-only");

const MODELS = {
    landmarker: {
        file: "face_landmarker.task",
        url: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    },
    buffalo: {
        url: "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
        members: { "det_500m.onnx": 2524817, "w600k_mbf.onnx": 13616099 },
    },
};

function log(...a) { console.log("[setup-assets]", ...a); }

function copyDir(src, dst, filter = () => true) {
    if (!existsSync(src)) { log(`skip (missing) ${src}`); return 0; }
    mkdirSync(dst, { recursive: true });
    let n = 0;
    for (const f of readdirSync(src)) {
        if (!filter(f)) continue;
        copyFileSync(join(src, f), join(dst, f));
        n++;
    }
    return n;
}

function copyRuntimes() {
    const ort = copyDir(join(ROOT, "node_modules/onnxruntime-web/dist"), join(PUBLIC, "ort"),
        f => f.startsWith("ort-wasm") && (f.endsWith(".wasm") || f.endsWith(".mjs")));
    const mp = copyDir(join(ROOT, "node_modules/@mediapipe/tasks-vision/wasm"), join(PUBLIC, "mediapipe/wasm"));
    log(`copied ${ort} onnxruntime-web files, ${mp} MediaPipe wasm files`);
}

async function download(url, dest) {
    log(`downloading ${url}`);
    const res = await fetch(url, { redirect: "follow" });
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
    const total = Number(res.headers.get("content-length") ?? 0);
    const tmp = dest + ".part";
    const out = createWriteStream(tmp);
    let got = 0, lastPct = -1;
    for await (const chunk of res.body) {
        out.write(chunk);
        got += chunk.length;
        const pct = total ? Math.floor((got / total) * 100) : -1;
        if (pct >= 0 && pct % 10 === 0 && pct !== lastPct) { process.stdout.write(`  ${pct}%`); lastPct = pct; }
    }
    await new Promise((r, j) => out.end(e => (e ? j(e) : r())));
    process.stdout.write("\n");
    renameSync(tmp, dest);
}

/** Minimal ZIP reader (stored + deflate) so no unzip tool is needed on Windows. */
function extractZipMembers(zipPath, wanted, outDir) {
    const buf = readFileSync(zipPath);
    let eocd = buf.length - 22;
    while (eocd >= 0 && buf.readUInt32LE(eocd) !== 0x06054b50) eocd--;
    if (eocd < 0) throw new Error("not a zip file");
    const count = buf.readUInt16LE(eocd + 10);
    let p = buf.readUInt32LE(eocd + 16);
    const found = [];
    for (let i = 0; i < count; i++) {
        const method = buf.readUInt16LE(p + 10);
        const csize = buf.readUInt32LE(p + 20);
        const nameLen = buf.readUInt16LE(p + 28), extraLen = buf.readUInt16LE(p + 30), commentLen = buf.readUInt16LE(p + 32);
        const local = buf.readUInt32LE(p + 42);
        const name = buf.subarray(p + 46, p + 46 + nameLen).toString("utf8");
        p += 46 + nameLen + extraLen + commentLen;
        const base = name.split("/").pop();
        if (!wanted.includes(base)) continue;
        const lNameLen = buf.readUInt16LE(local + 26), lExtraLen = buf.readUInt16LE(local + 28);
        const data = buf.subarray(local + 30 + lNameLen + lExtraLen, local + 30 + lNameLen + lExtraLen + csize);
        const content = method === 0 ? data : method === 8 ? inflateRawSync(data) : null;
        if (!content) throw new Error(`unsupported zip method ${method} for ${name}`);
        writeFileSync(join(outDir, base), content);
        found.push(base);
    }
    return found;
}

const sizeOk = (f, size) => existsSync(f) && (!size || statSync(f).size === size);

async function ensureModels() {
    const dir = join(PUBLIC, "models");
    mkdirSync(dir, { recursive: true });

    // 1) InsightFace detector + recogniser (reuse <repo>/models if the backend already downloaded them)
    for (const [f, s] of Object.entries(MODELS.buffalo.members)) {
        if (!sizeOk(join(dir, f), s) && sizeOk(join(REPO_MODELS, f), s)) {
            copyFileSync(join(REPO_MODELS, f), join(dir, f));
            log(`copied ${f} from ${REPO_MODELS}`);
        }
    }
    const still = Object.entries(MODELS.buffalo.members).filter(([f, s]) => !sizeOk(join(dir, f), s)).map(([f]) => f);
    if (still.length) {
        const zip = join(dir, "buffalo_s.zip");
        if (!existsSync(zip)) await download(MODELS.buffalo.url, zip);
        const got = extractZipMembers(zip, still, dir);
        log(`extracted ${got.join(", ")}`);
        rmSync(zip);
    }

    // 2) MediaPipe face landmarker (tracking + pose)
    const lm = join(dir, MODELS.landmarker.file);
    if (!existsSync(lm)) {
        try { await download(MODELS.landmarker.url, lm); }
        catch (e) { log(`WARNING could not download ${MODELS.landmarker.file} (${e.message}); the app will fetch it from Google's CDN at runtime`); }
    }

    for (const f of [...Object.keys(MODELS.buffalo.members), MODELS.landmarker.file]) {
        if (!existsSync(join(dir, f))) continue;
        const h = createHash("sha256").update(readFileSync(join(dir, f))).digest("hex").slice(0, 16);
        log(`ok ${f.padEnd(22)} ${(statSync(join(dir, f)).size / 1e6).toFixed(1)} MB  sha256:${h}…`);
    }
}

try {
    copyRuntimes();
    if (!runtimeOnly) await ensureModels();
    else if (!existsSync(join(PUBLIC, "models", "w600k_mbf.onnx"))) log("models not downloaded yet — run `npm run setup:assets`");
} catch (e) {
    console.error("[setup-assets] failed:", e.message);
    process.exit(runtimeOnly ? 0 : 1); // never break `npm install`
}
