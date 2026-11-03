#!/usr/bin/env node
/**
 * Removes files from out/ (the static export) that the browser never loads, so the site is smaller and
 * every file fits static hosts' per-file limits (Cloudflare Pages: 25 MiB).
 *   - models/arcface/w600k_mbf.onnx: the fp32 original; the browser loads the fp16 copy
 *   - ONNX Runtime WebGPU/JSPI/asyncify builds, in ort/ and _next/static/media, unless NEXT_PUBLIC_EMBEDDING_WEBGPU=true
 *     (the default "onnxruntime-web/wasm" bundle only loads ort-wasm-simd-threaded.{wasm,mjs})
 *
 * Usage: node scripts/prune-static-export.mjs [outDir]   (run by `npm run build:static`)
 */
import { existsSync, readdirSync, rmSync, statSync } from "node:fs";
import path from "node:path";

const out = path.resolve(process.argv[2] ?? "out");
if (!existsSync(out)) {
    console.error(`${out} does not exist; run the static build first`);
    process.exit(1);
}
const webgpu = process.env.NEXT_PUBLIC_EMBEDDING_WEBGPU === "true";
const removed = [];

const remove = rel => {
    const file = path.join(out, rel);
    if (existsSync(file)) {
        removed.push([rel, statSync(file).size]);
        rmSync(file);
    }
};

remove("models/arcface/w600k_mbf.onnx");
if (!webgpu) {
    // public/ort copies, plus the hashed copy the bundler emits for the (unused) WebGPU import
    for (const dir of ["ort", "_next/static/media"]) {
        if (!existsSync(path.join(out, dir))) continue;
        for (const f of readdirSync(path.join(out, dir))) {
            if (/\.(jsep|jspi|asyncify)\./.test(f)) remove(path.join(dir, f));
        }
    }
}

const mb = n => `${(n / 1024 / 1024).toFixed(1)} MB`;
for (const [rel, size] of removed) console.log(`  removed ${rel} (${mb(size)})`);

// anything still over the strictest common per-file limit would make the deploy fail
const LIMIT = 25 * 1024 * 1024;
const tooBig = [];
const walk = dir => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const p = path.join(dir, entry.name);
        if (entry.isDirectory()) walk(p);
        else if (statSync(p).size > LIMIT) tooBig.push(path.relative(out, p));
    }
};
walk(out);
if (tooBig.length) console.warn(`  warning: over 25 MiB, some static hosts refuse these: ${tooBig.join(", ")}`);
console.log(`Static export ready in ${path.relative(process.cwd(), out) || "."}`);
