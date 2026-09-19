import { spawn, spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const port = 5273;
const samples = Math.min(
  Math.max(Number(process.env.KVT_GRID_SAMPLES ?? 105), 1),
  105,
);
const run = `browser-grid-${new Date()
  .toISOString()
  .replace(/[-:.TZ]/g, "")
  .slice(0, 17)}`;

function browserPath() {
  const candidates = [
    process.env.CHROME,
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
  ].filter((path) => typeof path === "string");
  for (const candidate of candidates) {
    if (spawnSync("test", ["-x", candidate]).status === 0) return candidate;
  }
  throw new Error(
    "Chrome or Chromium was not found. Set CHROME to a headless browser executable.",
  );
}

async function stop(process) {
  if (process.exitCode === null && process.signalCode === null)
    process.kill("SIGTERM");
  if (process.exitCode !== null || process.signalCode !== null) return;
  await new Promise((resolve) => process.once("exit", resolve));
}

async function report(path, browser) {
  for (let attempt = 0; attempt < 1200; attempt += 1) {
    try {
      return JSON.parse(await readFile(path, "utf8"));
    } catch (error) {
      if (browser.exitCode !== null || browser.signalCode !== null)
        throw new Error(
          "headless browser exited before writing its evaluation report",
        );
      if (!(error instanceof Error) || !error.message.includes("ENOENT"))
        throw error;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("browser grid evaluation timed out");
}

const profile = await mkdtemp(join(tmpdir(), "kvt-grid-chrome-"));
let headless;
try {
  const browser = browserPath();
  headless = spawn(
    browser,
    [
      "--headless=new",
      "--disable-gpu",
      `--user-data-dir=${profile}`,
      `http://localhost:${port}/grid-eval.html?samples=${samples}&run=${encodeURIComponent(run)}`,
    ],
    { stdio: "ignore" },
  );
  const output = await report(
    join(root, "data", "evaluations", `${run}.json`),
    headless,
  );
  if (output.error) throw new Error(output.error);
  console.log(
    JSON.stringify(
      {
        report: join(root, "data", "evaluations", `${run}.json`),
        summary: output.summary,
      },
      null,
    ),
  );
} finally {
  if (headless) await stop(headless);
  await rm(profile, { recursive: true, force: true });
}
