import { spawn, spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const webRoot = new URL(".", import.meta.url).pathname;
const root = new URL("..", import.meta.url).pathname;
const port = 5274;
const samples = Math.min(
  Math.max(Number(process.env.KVT_EVAL_SAMPLES ?? 12), 1),
  100,
);
const run = `calibration-${new Date()
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
    const result = spawnSync("test", ["-x", candidate]);
    if (result.status === 0) return candidate;
  }
  for (const command of ["google-chrome", "chromium", "chromium-browser"]) {
    const result = spawnSync("which", [command], { encoding: "utf8" });
    if (result.status === 0) return result.stdout.trim();
  }
  throw new Error(
    "Chrome or Chromium was not found. Set CHROME to a headless browser executable.",
  );
}

async function waitForServer() {
  const url = `http://127.0.0.1:${port}/eval.html`;
  for (let attempt = 0; attempt < 60; attempt += 1) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // Vite is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`Vite did not start on port ${port}`);
}

function waitForExit(process) {
  if (process.exitCode !== null || process.signalCode !== null)
    return Promise.resolve();
  return new Promise((resolve) => process.once("exit", resolve));
}

async function stop(process) {
  if (process.exitCode === null && process.signalCode === null)
    process.kill("SIGTERM");
  await waitForExit(process);
}

async function waitForReport(reportPath, browser) {
  for (let attempt = 0; attempt < 600; attempt += 1) {
    try {
      return JSON.parse(await readFile(reportPath, "utf8"));
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
  throw new Error("evaluation timed out before writing its report");
}

const browser = browserPath();
const profile = await mkdtemp(join(tmpdir(), "kvt-calibration-chrome-"));
const server = spawn(
  "bun",
  ["run", "dev", "--", "--host", "127.0.0.1", "--port", String(port)],
  {
    cwd: webRoot,
    stdio: "ignore",
  },
);
let headless;
try {
  await waitForServer();
  const url = `http://127.0.0.1:${port}/eval.html?samples=${samples}&run=${encodeURIComponent(run)}`;
  headless = spawn(
    browser,
    ["--headless=new", "--disable-gpu", `--user-data-dir=${profile}`, url],
    { stdio: "ignore" },
  );
  const reportPath = join(root, "data", "evaluations", `${run}.json`);
  const report = await waitForReport(reportPath, headless);
  if (report.error) throw new Error(report.error);
  console.log(
    JSON.stringify({ report: reportPath, tracks: report.tracks }, null, 2),
  );
} finally {
  if (headless) await stop(headless);
  await stop(server);
  await rm(profile, { recursive: true, force: true });
}
