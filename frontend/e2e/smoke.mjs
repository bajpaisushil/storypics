// End-to-end browser test for StoryPix, driving the locally installed Google Chrome (no browser download).
//
//   node e2e/smoke.mjs ui     # page renders, form works, generator status is reported honestly
//   node e2e/smoke.mjs full   # also generates a real storybook (needs the image model) and regenerates a scene
//
// Env: E2E_URL (default http://localhost:3000), E2E_API (default http://localhost:8000), E2E_OUT (screenshots dir).
import { mkdirSync } from "node:fs";
import { chromium } from "playwright-core";

const MODE = process.argv[2] ?? "ui";
const BASE = process.env.E2E_URL ?? "http://localhost:3000";
const API = process.env.E2E_API ?? "http://localhost:8000";
const OUT = process.env.E2E_OUT ?? "e2e/screenshots";
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ channel: "chrome", headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
const page = await context.newPage();
const consoleErrors = [];
page.on("console", (m) => m.type() === "error" && consoleErrors.push(m.text()));
page.on("pageerror", (e) => consoleErrors.push(String(e)));

const shot = (name, opts = {}) => page.screenshot({ path: `${OUT}/${name}.png`, ...opts });
const step = (msg) => console.log(`• ${msg}`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function apiJson(path) {
  const r = await fetch(`${API}${path}`);
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

try {
  step(`open ${BASE}`);
  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.getByRole("button", { name: /Generate Story/i }).waitFor({ timeout: 30_000 });
  // wait until React has hydrated (the health check has run), otherwise early clicks are lost
  await page.waitForFunction(() => document.querySelector('[aria-label="5 scenes"]') !== null, null, { timeout: 30_000 });
  await sleep(1500);
  await shot("01-home");

  step("fill the example story and choose options");
  await page.getByRole("button", { name: /Try an example/i }).click();
  const text = await page.getByPlaceholder(/Once upon a time/i).inputValue();
  if (!text.includes("hen")) throw new Error("example story was not inserted");
  await page.getByText("Watercolor", { exact: true }).first().click();
  await page.getByLabel("5 scenes").click();
  await page.getByLabel("Aspect ratio 4:3").click();
  if ((await page.getByLabel("5 scenes").getAttribute("data-state")) !== "on") throw new Error("scene count not selected");
  await shot("02-form-filled");

  const health = await apiJson("/api/health");
  console.log(`  generator available: ${health.generator.available} (${health.generator.reason ?? "ok"})`);
  if (!health.generator.available) {
    const button = page.getByRole("button", { name: /Generate Story/i });
    if (await button.isEnabled()) throw new Error("Generate must be disabled while the generator is unavailable");
    step("generator unavailable → Generate is disabled and the reason is shown");
    await shot("03-generator-unavailable", { fullPage: true });
  }

  if (MODE === "full") {
    if (!health.generator.available) throw new Error("full mode needs the image model");
    step("generate the storybook");
    await page.getByRole("button", { name: /Generate Story/i }).click();
    await page.waitForURL(/[?&]job=/, { timeout: 60_000 });
    const jobId = new URL(page.url()).searchParams.get("job");
    console.log(`  job ${jobId}`);
    await page.getByLabel("Overall progress").waitFor({ timeout: 60_000 });
    await sleep(4000);
    await shot("04-generating");

    let job;
    const started = Date.now();
    let midShot = false;
    for (;;) {
      job = await apiJson(`/api/jobs/${jobId}`);
      const done = job.scenes.filter((s) => s.status === "completed").length;
      if (!midShot && done >= 2) {
        await shot("05-partial");
        midShot = true;
      }
      if (["completed", "partial", "failed", "cancelled"].includes(job.status)) break;
      if (Date.now() - started > 45 * 60_000) throw new Error("generation took longer than 45 minutes");
      await sleep(5000);
    }
    console.log(`  finished: ${job.status} in ${Math.round((Date.now() - started) / 1000)} s`);
    if (job.status !== "completed") throw new Error(`job ended as ${job.status}: ${job.error ?? ""}`);
    await sleep(2000);
    await shot("06-storybook", { fullPage: true });

    step("downloads");
    for (const path of [`/api/jobs/${jobId}/export.zip`, `/api/jobs/${jobId}/export.pdf`,
                        `/api/jobs/${jobId}/scenes/1/download`]) {
      const r = await fetch(`${API}${path}`);
      console.log(`  ${path} → ${r.status} ${r.headers.get("content-type")} ${(await r.arrayBuffer()).byteLength} bytes`);
      if (!r.ok) throw new Error(`download failed: ${path}`);
    }

    step("regenerate scene 2 with an instruction, keeping the composition");
    // the viewer follows generation to the last scene; pick scene 2 from the thumbnail strip
    await page.getByRole("navigation", { name: "Scenes" }).getByRole("button", { name: /^Scene 2:/ }).click();
    await page.getByRole("button", { name: /^Regenerate$/ }).first().click();
    await page.getByRole("dialog").getByRole("textbox").fill("Make the hen happier");
    await page.getByRole("dialog").getByText("Keep composition").click();
    await shot("07-regenerate-dialog");
    await page.getByRole("dialog").getByRole("button", { name: /Regenerate/ }).click();
    for (;;) {
      job = await apiJson(`/api/jobs/${jobId}`);
      const scene = job.scenes.find((s) => s.index === 2);
      if (scene.versions.length >= 2 && scene.status === "completed") break;
      if (scene.status === "failed") throw new Error(`regeneration failed: ${scene.error}`);
      await sleep(4000);
    }
    await sleep(2000);
    await shot("08-regenerated", { fullPage: true });

    step("mobile layout");
    await page.setViewportSize({ width: 390, height: 844 });
    await sleep(1000);
    await shot("09-mobile", { fullPage: true });
  }

  const relevant = consoleErrors.filter((e) => !/favicon|ERR_CONNECTION_REFUSED|Failed to load resource/.test(e));
  if (relevant.length) {
    console.log("console errors:\n  " + relevant.join("\n  "));
    throw new Error(`${relevant.length} console error(s)`);
  }
  console.log(`✓ e2e ${MODE} passed — screenshots in ${OUT}/`);
} catch (err) {
  await shot("zz-failure", { fullPage: true }).catch(() => {});
  console.error(`✗ e2e failed: ${err.message}`);
  process.exitCode = 1;
} finally {
  await browser.close();
}
