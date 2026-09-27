import { spawn, spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer } from "vite";

const webRoot = new URL(".", import.meta.url).pathname;
const root = new URL("..", import.meta.url).pathname;
const samples = Math.min(
  Math.max(Number(process.env.PIANOCV_GRID_SAMPLES ?? 105), 1),
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

// a shared dev server may already be serving a different checkout on the usual port, so
// this runner owns a throwaway vite server on an OS-assigned port instead of reusing it
const server = await createServer({
  root: webRoot,
  logLevel: "silent",
  clearScreen: false,
  server: { host: "127.0.0.1", port: 0, strictPort: false },
});
await server.listen();
const address = server.httpServer?.address();
if (!address || typeof address === "string")
  throw new Error("vite dev server did not bind a port");
const port = address.port;

const profile = await mkdtemp(join(tmpdir(), "pianocv-grid-chrome-"));
let headless;
try {
  const browser = browserPath();
  headless = spawn(
    browser,
    [
      "--headless=new",
      "--disable-gpu",
      "--no-sandbox",
      `--user-data-dir=${profile}`,
      `http://127.0.0.1:${port}/grid-eval.html?samples=${samples}&run=${encodeURIComponent(run)}`,
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
        byElevation: output.byElevation,
        byAbsAzimuth: output.byAbsAzimuth,
      },
      null,
      2,
    ),
  );
} finally {
  if (headless) await stop(headless);
  await rm(profile, { recursive: true, force: true });
  await server.close();
}
