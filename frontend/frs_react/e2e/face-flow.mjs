/**
 * End-to-end smoke test of the on-device pipeline in a real Chromium with a fake webcam.
 *
 *   1. backend:  cd backend/frs/src && QDRANT_LOCATION=:memory: FRS_ENABLE_SERVER_INFERENCE=false uvicorn api.server:app --port 5000
 *   2. web:      npm run build && npm start
 *   3. video:    ffmpeg -loop 1 -i ../../backend/frs/samples/480.jpg -vf "scale=480:480,pad=640:480:80:0:white,format=yuv420p" -t 2 -r 15 /tmp/face.y4m
 *   4. run:      FAKE_VIDEO=/tmp/face.y4m node e2e/face-flow.mjs        (needs `npm i -D playwright` or a global playwright)
 *
 * Registers the face with the quick (frontal only) plan, logs in with the same face and prints the
 * pipeline timings reported by /examples/pipeline.
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const BASE = process.env.BASE_URL ?? "http://localhost:3000";
const video = process.env.FAKE_VIDEO;
if (!video) {
    console.error("set FAKE_VIDEO to a .y4m / .mjpeg file");
    process.exit(2);
}

const browser = await chromium.launch({
    args: [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        `--use-file-for-fake-video-capture=${video}`,
        "--enable-unsafe-webgpu",
    ],
});
const context = await browser.newContext({ permissions: ["camera"] });
const page = await context.newPage();
page.on("console", m => { if (["error", "warning"].includes(m.type())) console.log(`[browser ${m.type()}] ${m.text()}`); });

const step = async (name, fn) => {
    const t = Date.now();
    await fn();
    console.log(`ok  ${name} (${((Date.now() - t) / 1000).toFixed(1)}s)`);
};

try {
    await step("quick registration", async () => {
        await page.goto(`${BASE}/face/registration?plan=quick&fullname=E2E%20User&email=e2e@example.com`);
        await page.getByText("Face enrolled successfully!").waitFor({ timeout: 90_000 });
    });

    await step("face login", async () => {
        await page.goto(`${BASE}/face/login`);
        await page.getByText("Welcome, E2E User!").waitFor({ timeout: 60_000 });
        await page.waitForURL(/\/dashboard\?/, { timeout: 15_000 });
    });

    await step("pipeline inspector", async () => {
        await page.goto(`${BASE}/examples/pipeline`);
        const cell = page.locator("tr", { hasText: "Embedding incl. flip TTA" }).locator("td").nth(1);
        await page.waitForFunction(el => el && /\d/.test(el.textContent ?? ""), await cell.elementHandle({ timeout: 90_000 }), { timeout: 90_000 });
        await page.waitForTimeout(3000);
        for (const row of await page.locator("table tr").all()) {
            console.log("    " + (await row.innerText()).replace(/\s+/g, " "));
        }
    });
} finally {
    await browser.close();
}
