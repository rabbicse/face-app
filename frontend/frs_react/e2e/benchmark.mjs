/**
 * Web pipeline benchmark: cold + warm start of /examples/pipeline with a fake webcam.
 * Reports bytes downloaded per asset, model init times, median per-stage timings and the time until the
 * first embedding is shown. Same prerequisites as face-flow.mjs:
 *
 *   FAKE_VIDEO=/tmp/face.y4m node e2e/benchmark.mjs [--json]
 *
 * Headless Chromium renders WebGL in software and has no WebGPU, so absolute numbers are pessimistic;
 * compare runs on the same machine.
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const BASE = process.env.BASE_URL ?? "http://localhost:3000";
const SAMPLES = Number(process.env.SAMPLES ?? 12);

const browser = await chromium.launch({
    args: [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        `--use-file-for-fake-video-capture=${process.env.FAKE_VIDEO}`,
        "--enable-unsafe-webgpu",
    ],
});
const context = await browser.newContext({ permissions: ["camera"] });
const page = await context.newPage();

// context-level events also cover the Web Worker (ONNX Runtime wasm + model); cache hits count as 0
const bytes = new Map();
context.on("requestfinished", async req => {
    const res = await req.response();
    if (!res || res.fromServiceWorker()) return;
    const sizes = await req.sizes();
    const fromCache = res.status() === 304 || sizes.responseBodySize === 0;
    bytes.set(req.url(), (bytes.get(req.url()) ?? 0) + (fromCache ? 0 : sizes.responseBodySize + sizes.responseHeadersSize));
});

async function table() {
    const rows = await page.locator("table tr").allInnerTexts();
    return Object.fromEntries(rows.map(r => r.split("\t").map(s => s.trim())));
}

const num = s => Number.parseFloat(String(s));
const median = a => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];

async function run(label) {
    bytes.clear();
    const t0 = Date.now();
    await page.goto(`${BASE}/examples/pipeline`);
    const embedCell = page.locator("tr", { hasText: "Embedding incl. flip TTA" }).locator("td").nth(1);
    await embedCell.filter({ hasText: /\d/ }).waitFor({ timeout: 120_000 });
    const firstEmbeddingMs = Date.now() - t0;

    const samples = [];
    for (let i = 0; i < SAMPLES; i++) {
        await page.waitForTimeout(400);
        samples.push(await table());
    }
    const last = samples[samples.length - 1];
    const [lmInit, embInit] = last["Model init: landmarker / embedder (ms)"].split("/").map(num);
    const total = [...bytes.values()].reduce((a, b) => a + b, 0);
    const heavy = [...bytes.entries()].filter(([, b]) => b > 200_000)
        .map(([u, b]) => [new URL(u).pathname, Math.round(b / 1024)]);
    return {
        label,
        backend: last["Landmarker / embedding backend"],
        downloadedKB: Math.round(total / 1024),
        largestKB: Object.fromEntries(heavy),
        landmarkerInitMs: lmInit,
        embedderInitMs: embInit,
        firstEmbeddingMs,
        fps: median(samples.map(s => num(s["Camera FPS processed"]))),
        landmarksMs: median(samples.map(s => num(s["Landmarks (ms)"]))),
        alignMs: median(samples.map(s => num(s["Align + quality (ms)"]))),
        embedMs: median(samples.map(s => num(s["Embedding incl. flip TTA (ms)"]))),
    };
}

try {
    const cold = await run("cold");
    const warm = await run("warm (HTTP cache)");
    if (process.argv.includes("--json")) {
        console.log(JSON.stringify([cold, warm], null, 1));
    } else {
        for (const r of [cold, warm]) {
            console.log(`\n== ${r.label}`);
            for (const [k, v] of Object.entries(r)) if (k !== "label") console.log(`  ${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`);
        }
    }
} finally {
    await browser.close();
}
