/**
 * End-to-end smoke test of the on-device pipeline in a real Chromium with a fake webcam.
 *
 *   1. backend:  cd backend/frs && QDRANT_LOCATION=:memory: uvicorn main:app --port 5000
 *   2. web:      npm run build && npm start
 *   3. video:    ffmpeg -loop 1 -i ../../backend/frs/samples/480.jpg -vf "scale=480:480,pad=640:480:80:0:white,noise=alls=25:allf=t,format=yuv420p" -t 2 -r 15 /tmp/face.y4m
 *                (the noise matters: enrolment rejects near-identical samples, so a frozen image never finishes)
 *   4. run:      FAKE_VIDEO=/tmp/face.y4m BLINK_VIDEO=/tmp/blink.y4m node e2e/face-flow.mjs
 *                (needs `npm i -D playwright` or a global playwright)
 *
 * Registers the face with the quick (frontal only) plan, checks that the still photo is refused at login
 * (liveness: no blink), logs in with BLINK_VIDEO (the same photo with a few closed-eye frames) and prints the
 * pipeline timings reported by /examples/pipeline. Without BLINK_VIDEO, build the app with
 * NEXT_PUBLIC_REQUIRE_BLINK=false and the login is done with FAKE_VIDEO.
 */
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const BASE = process.env.BASE_URL ?? "http://localhost:3000";
const video = process.env.FAKE_VIDEO;
const blinkVideo = process.env.BLINK_VIDEO;
if (!video) {
    console.error("set FAKE_VIDEO to a .y4m / .mjpeg file");
    process.exit(2);
}

/** the fake camera is chosen at launch, so every video gets its own browser */
const browsers = [];
async function openPage(videoFile) {
    const browser = await chromium.launch({
        args: [
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            `--use-file-for-fake-video-capture=${videoFile}`,
            "--enable-unsafe-webgpu",
        ],
    });
    browsers.push(browser);
    const page = await (await browser.newContext({ permissions: ["camera"] })).newPage();
    page.on("console", m => { if (m.type() === "error") console.log(`[browser error] ${m.text()}`); });
    return page;
}

const page = await openPage(video);

const step = async (name, fn) => {
    const t = Date.now();
    await fn();
    console.log(`ok  ${name} (${((Date.now() - t) / 1000).toFixed(1)}s)`);
};

try {
    await step("quick registration", async () => {
        await page.goto(`${BASE}/face/registration?plan=quick&fullname=E2E%20User&email=e2e@example.com`);
        await page.getByRole("button", { name: "I agree, turn on camera" }).click();
        await page.getByText("Face enrolled successfully!").waitFor({ timeout: 90_000 });
    });

    if (blinkVideo) {
        await step("still photo is refused (no blink)", async () => {
            await page.goto(`${BASE}/face/login`);
            await page.getByRole("button", { name: "Turn on camera" }).click();
            await page.getByText("Please blink to continue").first().waitFor({ timeout: 60_000 });
            await page.waitForTimeout(8000);
            if (await page.getByText("Welcome, E2E User!").isVisible()) throw new Error("photo was accepted");
        });
    }

    await step("face login", async () => {
        const loginPage = blinkVideo ? await openPage(blinkVideo) : page;
        await loginPage.goto(`${BASE}/face/login`);
        await loginPage.getByRole("button", { name: "Turn on camera" }).click();
        await loginPage.getByText("Welcome, E2E User!").waitFor({ timeout: 60_000 });
        await loginPage.waitForURL(/\/dashboard$/, { timeout: 15_000 });
        await loginPage.getByRole("heading", { name: "Signed in as E2E User" }).waitFor({ timeout: 15_000 });
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
    await Promise.all(browsers.map(b => b.close()));
}
