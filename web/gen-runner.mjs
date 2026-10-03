import { spawn, spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer } from "vite";

const webRoot = new URL(".", import.meta.url).pathname;
const root = new URL("..", import.meta.url).pathname;
const targetFrames = Math.max(
  1,
  Number(process.env.PIANOCV_GEN_FRAMES) || 8000,
);
const motion = process.env.PIANOCV_GEN_MOTION === "1";
const outDirName =
  process.env.PIANOCV_GEN_OUT || (motion ? "synth-motion" : "synth-case");

const POLL_MS = 1_000;
const STALL_MS = 60_000;
const PROGRESS_STEP = 500;
const EVAL_TIMEOUT_MS = 5_000;

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

// chrome writes the debugger's real port here once devtools is listening, so a
// remote-debugging-port of 0 (an OS-assigned port) can still be discovered
async function waitForDevtoolsPort(profile, browser) {
  const path = join(profile, "DevToolsActivePort");
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try {
      const content = await readFile(path, "utf8");
      const port = Number(content.split("\n")[0]);
      if (Number.isInteger(port) && port > 0) return port;
    } catch (error) {
      if (browser.exitCode !== null || browser.signalCode !== null)
        throw new Error("headless browser exited before opening devtools");
      if (!(error instanceof Error) || !error.message.includes("ENOENT"))
        throw error;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error("headless browser never opened a devtools port");
}

function createCdpClient(webSocketDebuggerUrl) {
  const ws = new WebSocket(webSocketDebuggerUrl);
  const pending = new Map();
  let nextId = 0;
  const fail = (error) => {
    for (const { reject } of pending.values()) reject(error);
    pending.clear();
  };
  ws.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    const waiting = pending.get(message.id);
    if (!waiting) return;
    pending.delete(message.id);
    waiting.resolve(message);
  });
  ws.addEventListener("close", () => fail(new Error("devtools socket closed")));
  ws.addEventListener("error", () => fail(new Error("devtools socket error")));
  const opened = new Promise((resolve, reject) => {
    ws.addEventListener("open", () => resolve(), { once: true });
    ws.addEventListener(
      "error",
      () => reject(new Error("devtools socket failed to open")),
      { once: true },
    );
  });
  const send = async (method, params = {}) => {
    await opened;
    nextId += 1;
    const id = nextId;
    const message = new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
    });
    ws.send(JSON.stringify({ id, method, params }));
    const timeout = new Promise((_, reject) => {
      setTimeout(() => {
        pending.delete(id);
        reject(new Error(`${method} timed out`));
      }, EVAL_TIMEOUT_MS);
    });
    return Promise.race([message, timeout]);
  };
  return { send, close: () => ws.close() };
}

// gen.ts's status readout ends in "saved N" or "saved N/target"; this is the one number the
// page exposes for an unattended runner to poll, so no extra window global is needed
function savedCountFrom(pageText) {
  const match =
    typeof pageText === "string" ? /saved (\d+)/.exec(pageText) : null;
  return match ? Number(match[1]) : null;
}

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
const vitePort = address.port;

const profile = await mkdtemp(join(tmpdir(), "pianocv-gen-chrome-"));
let headless;
let cdp;
try {
  const browser = browserPath();
  const url = `http://127.0.0.1:${vitePort}/gen.html?out=${outDirName}&frames=${targetFrames}${motion ? "&motion=1" : ""}${process.env.PIANOCV_GEN_POSE ? `&${process.env.PIANOCV_GEN_POSE}` : ""}`;
  headless = spawn(
    browser,
    [
      "--headless=new",
      "--disable-gpu",
      // gen.html renders through three.js's WebGLRenderer, which a headless GPU-less host can
      // only satisfy through software rendering
      "--enable-unsafe-swiftshader",
      "--no-sandbox",
      "--remote-debugging-port=0",
      `--user-data-dir=${profile}`,
      url,
    ],
    { stdio: "ignore" },
  );
  const devtoolsPort = await waitForDevtoolsPort(profile, headless);
  const targets = await fetch(
    `http://127.0.0.1:${devtoolsPort}/json/list`,
  ).then((response) => response.json());
  const target = targets.find((t) => t.type === "page") ?? targets[0];
  if (!target)
    throw new Error("headless browser opened no debuggable page target");
  cdp = createCdpClient(target.webSocketDebuggerUrl);
  await cdp.send("Page.enable");
  await cdp.send("Runtime.enable");

  const outDir = join(root, "data", outDirName);
  console.log(`gen-runner: rendering ${targetFrames} frames into ${outDir}`);

  let saved = 0;
  let lastPrinted = 0;
  let lastChangeAt = Date.now();
  while (saved < targetFrames) {
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
    // a slow poll under system load is not the same as no progress; only the stall check below,
    // which looks at real elapsed time since the last save, should ever abort the run
    let response;
    try {
      response = await cdp.send("Runtime.evaluate", {
        expression: "document.body.innerText",
        returnByValue: true,
      });
    } catch (error) {
      if (Date.now() - lastChangeAt > STALL_MS) throw error;
      continue;
    }
    const count = savedCountFrom(response?.result?.result?.value);
    if (count !== null && count > saved) {
      saved = count;
      lastChangeAt = Date.now();
      if (saved - lastPrinted >= PROGRESS_STEP) {
        lastPrinted = saved - (saved % PROGRESS_STEP);
        console.log(`gen-runner: saved ${saved}/${targetFrames}`);
      }
      continue;
    }
    if (Date.now() - lastChangeAt > STALL_MS) {
      throw new Error(
        `gen-runner: saving stalled for ${STALL_MS / 1000}s at ${saved}/${targetFrames} frames`,
      );
    }
  }
  console.log(`gen-runner: done, saved ${saved} frames to ${outDir}`);
} finally {
  if (cdp) cdp.close();
  if (headless) await stop(headless);
  await rm(profile, { recursive: true, force: true });
  await server.close();
}
