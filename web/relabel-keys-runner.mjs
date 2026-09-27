import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import {
  cp,
  mkdir,
  mkdtemp,
  readdir,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer } from "vite";

const webRoot = new URL(".", import.meta.url).pathname;
const root = new URL("..", import.meta.url).pathname;
const toolsDir = join(root, "tools");
const dataDir = join(root, "data");
const realSeg2Dir = join(dataDir, "real-seg2");
const recordingsDir = join(dataDir, "recordings");
const outDir = join(dataDir, "real-seg2-keys");
const truthPath = join(dataDir, "recordings-keys-truth.json");

const run = `relabel-keys-${new Date()
  .toISOString()
  .replace(/[-:.TZ]/g, "")
  .slice(0, 17)}`;

const UNSAFE_NAME = /[^a-zA-Z0-9._-]/g;

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

async function waitForReport(path, browser) {
  for (let attempt = 0; attempt < 2400; attempt += 1) {
    try {
      return JSON.parse(await readFile(path, "utf8"));
    } catch (error) {
      if (browser.exitCode !== null || browser.signalCode !== null)
        throw new Error(
          "headless browser exited before writing its relabel report",
        );
      if (!(error instanceof Error) || !error.message.includes("ENOENT"))
        throw error;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("browser relabel timed out");
}

// mirrors web/lab-server.ts's shape for the corpora this task reads and writes, without
// touching that file while other work is in flight on it
function relabelServer(reportPath) {
  return {
    name: "pianocv-relabel-server",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url ?? "";
        if (
          req.method === "GET" &&
          url === "/relabel-lab/real-seg2/corners.json"
        ) {
          readFile(join(realSeg2Dir, "corners.json"))
            .then((body) => {
              res.setHeader("content-type", "application/json");
              res.end(body);
            })
            .catch(() => {
              res.statusCode = 404;
              res.end();
            });
          return;
        }
        const frame = /^\/relabel-lab\/real-seg2\/frame\/([^/?]+)$/.exec(url);
        if (req.method === "GET" && frame) {
          readFile(
            join(realSeg2Dir, "frames", frame[1].replace(UNSAFE_NAME, "")),
          )
            .then((body) => {
              res.setHeader("content-type", "image/png");
              res.end(body);
            })
            .catch(() => {
              res.statusCode = 404;
              res.end();
            });
          return;
        }
        if (
          req.method === "GET" &&
          url === "/relabel-lab/recordings/list.json"
        ) {
          listRecordingsWithCorners()
            .then((stems) => {
              res.setHeader("content-type", "application/json");
              res.end(JSON.stringify(stems));
            })
            .catch(() => {
              res.statusCode = 500;
              res.end();
            });
          return;
        }
        const sidecar = /^\/relabel-lab\/recordings\/sidecar\/([^/?]+)$/.exec(
          url,
        );
        if (req.method === "GET" && sidecar) {
          readFile(join(recordingsDir, sidecar[1].replace(UNSAFE_NAME, "")))
            .then((body) => {
              res.setHeader("content-type", "application/json");
              res.end(body);
            })
            .catch(() => {
              res.statusCode = 404;
              res.end();
            });
          return;
        }
        const video = /^\/relabel-lab\/recordings\/video\/([^/?]+)$/.exec(url);
        if (req.method === "GET" && video) {
          readFile(join(recordingsDir, video[1].replace(UNSAFE_NAME, "")))
            .then((body) => {
              res.setHeader("content-type", "video/webm");
              res.end(body);
            })
            .catch(() => {
              res.statusCode = 404;
              res.end();
            });
          return;
        }
        const report = /^\/relabel-lab\/report\/([^/?]+)$/.exec(url);
        if (req.method === "POST" && report) {
          const chunks = [];
          req.on("data", (chunk) => chunks.push(chunk));
          req.on("end", () => {
            writeFile(reportPath, Buffer.concat(chunks))
              .then(() => {
                res.statusCode = 204;
                res.end();
              })
              .catch(() => {
                res.statusCode = 500;
                res.end();
              });
          });
          return;
        }
        next();
      });
    },
  };
}

async function listRecordingsWithCorners() {
  const entries = await readdir(recordingsDir);
  const stems = [];
  for (const entry of entries) {
    if (!entry.endsWith(".webm")) continue;
    const stem = entry.slice(0, -".webm".length);
    const sidecarPath = join(recordingsDir, `${stem}.json`);
    if (!existsSync(sidecarPath)) continue;
    const sidecar = JSON.parse(await readFile(sidecarPath, "utf8"));
    if (Array.isArray(sidecar.corners) && sidecar.corners.length === 4)
      stems.push(stem);
  }
  return stems.sort();
}

function summarizeTrim(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return {
    n: sorted.length,
    min: sorted[0],
    median: sorted[Math.floor(sorted.length / 2)],
    max: sorted.at(-1),
  };
}

async function buildRealSeg2Keys(report) {
  const corners = JSON.parse(
    await readFile(join(realSeg2Dir, "corners.json"), "utf8"),
  );
  const labels = JSON.parse(
    await readFile(join(realSeg2Dir, "labels.json"), "utf8"),
  );
  const splits = JSON.parse(
    await readFile(join(realSeg2Dir, "splits.json"), "utf8"),
  );

  const perFrameReport = {};
  const keptCorners = {};
  const droppedBySplitReason = {};
  const leftEdges = [];
  const rightEdges = [];

  for (const stem of Object.keys(corners)) {
    const result = report.realSeg2[stem];
    if (result?.status !== "trimmed") {
      const reason = !result
        ? "missing from browser report"
        : result.status === "unresolved"
          ? "corner order could not be resolved"
          : "far edge could not be measured";
      perFrameReport[stem] = { kept: false, reason };
      droppedBySplitReason[reason] = (droppedBySplitReason[reason] ?? 0) + 1;
      continue;
    }
    keptCorners[stem] = result.quad.map((p) => [p.x, p.y]);
    perFrameReport[stem] = {
      kept: true,
      trim: result.edge,
      whiteKeys: result.whiteKeys,
      phase: result.phase,
      confidence: result.confidence,
    };
    leftEdges.push(result.edge.left);
    rightEdges.push(result.edge.right);
  }

  const keptNames = new Set(
    Object.keys(keptCorners).map((stem) => `${stem}.png`),
  );
  const newSplits = {};
  for (const [name, members] of Object.entries(splits)) {
    newSplits[name] = members.filter((member) => keptNames.has(member));
  }
  const emptySplit = Object.entries(newSplits).find(
    ([, members]) => members.length === 0,
  );

  const perSplit = Object.fromEntries(
    Object.entries(splits).map(([name, members]) => [
      name,
      { total: members.length, kept: newSplits[name]?.length ?? 0 },
    ]),
  );

  if (emptySplit) {
    return {
      written: false,
      error: `split "${emptySplit[0]}" would be empty after filtering; data/real-seg2-keys was not written`,
      perSplit,
      droppedBySplitReason,
      trim: {
        left: summarizeTrim(leftEdges),
        right: summarizeTrim(rightEdges),
      },
    };
  }

  const newLabels = {};
  for (const name of keptNames) {
    const stem = name.slice(0, -".png".length);
    newLabels[name] = {
      source_stem: labels[name].source_stem,
      corners: keptCorners[stem],
    };
  }

  await rm(outDir, { recursive: true, force: true });
  await mkdir(join(outDir, "frames"), { recursive: true });
  await mkdir(join(outDir, "masks"), { recursive: true });
  for (const name of keptNames) {
    await cp(join(realSeg2Dir, "frames", name), join(outDir, "frames", name));
  }
  await writeFile(
    join(outDir, "corners.json"),
    `${JSON.stringify(keptCorners, null, 2)}\n`,
  );
  await writeFile(
    join(outDir, "labels.json"),
    `${JSON.stringify(newLabels, null, 2)}\n`,
  );
  await writeFile(
    join(outDir, "splits.json"),
    `${JSON.stringify(newSplits, null, 2)}\n`,
  );
  await writeFile(
    join(outDir, "report.json"),
    `${JSON.stringify(perFrameReport, null, 2)}\n`,
  );

  const python = spawnSync(
    "uv",
    [
      "run",
      "python",
      "-m",
      "pianocv.relabelkeys",
      "--dir",
      outDir,
      "--original-corners",
      join(realSeg2Dir, "corners.json"),
    ],
    { cwd: toolsDir, stdio: "inherit" },
  );
  if (python.status !== 0) {
    throw new Error(
      "pianocv.relabelkeys failed to draw masks and the contact sheet",
    );
  }

  return {
    written: true,
    kept: keptNames.size,
    dropped: Object.keys(corners).length - keptNames.size,
    perSplit,
    droppedBySplitReason,
    trim: { left: summarizeTrim(leftEdges), right: summarizeTrim(rightEdges) },
  };
}

async function writeRecordingsTruth(report) {
  await writeFile(truthPath, `${JSON.stringify(report.recordings, null, 2)}\n`);
  const stems = Object.keys(report.recordings);
  const withTruth = stems.filter(
    (stem) => !("skipped" in report.recordings[stem]),
  );
  return { total: stems.length, withTruth: withTruth.length };
}

const scratch = await mkdtemp(join(tmpdir(), "pianocv-relabel-"));
const reportPath = join(scratch, "report.json");

const server = await createServer({
  root: webRoot,
  logLevel: "silent",
  clearScreen: false,
  server: { host: "127.0.0.1", port: 0, strictPort: false },
  plugins: [relabelServer(reportPath)],
});
await server.listen();
const address = server.httpServer?.address();
if (!address || typeof address === "string")
  throw new Error("vite dev server did not bind a port");
const port = address.port;

const profile = await mkdtemp(join(tmpdir(), "pianocv-relabel-chrome-"));
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
      `http://127.0.0.1:${port}/relabel-keys.html?run=${encodeURIComponent(run)}`,
    ],
    { stdio: "ignore" },
  );
  const report = await waitForReport(reportPath, headless);
  if (report.error) throw new Error(report.error);

  const corpus = await buildRealSeg2Keys(report);
  const truth = await writeRecordingsTruth(report);

  console.log(
    JSON.stringify(
      {
        realSeg2Keys: corpus,
        recordingsKeysTruth: truth,
        contactSheet: corpus.written ? join(outDir, "contact-sheet.png") : null,
      },
      null,
      2,
    ),
  );
  if (!corpus.written) {
    process.exitCode = 1;
  }
} finally {
  if (headless) await stop(headless);
  await rm(profile, { recursive: true, force: true });
  await rm(scratch, { recursive: true, force: true });
  await server.close();
}
