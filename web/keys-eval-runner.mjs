import { spawn, spawnSync } from "node:child_process";
import {
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
const recordingsDir = join(root, "data", "recordings");
const run = `keys-eval-${new Date()
  .toISOString()
  .replace(/[-:.TZ]/g, "")
  .slice(0, 17)}`;

const POLL_MS = 100;
const CLIP_BUDGET_MS = 20_000;
const STABLE_MS = 2_000;
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

async function clipsToRun() {
  const requested = process.env.PIANOCV_CLIPS;
  if (requested === "none") {
    return [];
  }
  if (requested) {
    return requested
      .split(",")
      .map((name) => name.trim())
      .filter((name) => name.length > 0);
  }
  const entries = await readdir(recordingsDir);
  return entries.filter((name) => name.endsWith(".webm")).sort();
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
  const send = async (method, params = {}, timeoutMs = EVAL_TIMEOUT_MS) => {
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
      }, timeoutMs);
    });
    return Promise.race([message, timeout]);
  };
  return { send, close: () => ws.close() };
}

// runs inside the page: pulls the HUD's own readouts (the last match of a label wins, since
// the panel's buttons can repeat a readout's word before the readouts grid ever does) plus the
// live key read, and folds the black-key alignment check into one small JSON blob so no per-pixel
// strip data ever has to cross the devtools socket
const POLL_EXPRESSION = `(() => {
  const lines = document.body.innerText.split("\\n");
  const lastValueAfter = (label) => {
    let index = -1;
    for (let i = 0; i < lines.length; i += 1) {
      if (lines[i] === label) index = i;
    }
    return index >= 0 && index + 1 < lines.length ? lines[index + 1] : null;
  };
  const read = window.pianocvKeyRead ?? null;
  const strip = window.pianocvKeyStrip ?? null;
  let alignment = null;
  if (read && read.kind === "read" && strip) {
    const luminance = (at) =>
      0.299 * strip.data[at] + 0.587 * strip.data[at + 1] + 0.114 * strip.data[at + 2];
    const bandTop = Math.round(0.15 * strip.height);
    const bandBottom = Math.round(0.4 * strip.height);
    const perBar = [];
    for (let i = 0; i < read.totalKeys; i += 1) {
      const key = read.keyAt(i);
      if (!key || !key.black) continue;
      const xs = key.bar.map((p) => p.x);
      const x0 = Math.max(0, Math.round(Math.min(...xs)));
      const x1 = Math.min(strip.width - 1, Math.round(Math.max(...xs)));
      let sum = 0;
      let count = 0;
      for (let y = bandTop; y <= bandBottom; y += 1) {
        for (let x = x0; x <= x1; x += 1) {
          sum += luminance((y * strip.width + x) * 4);
          count += 1;
        }
      }
      if (count > 0) perBar.push(sum / count);
    }
    const groups = 6;
    const sums = new Array(groups).fill(0);
    const counts = new Array(groups).fill(0);
    perBar.forEach((value, i) => {
      const group = Math.min(groups - 1, Math.floor((i * groups) / perBar.length));
      sums[group] += value;
      counts[group] += 1;
    });
    const refY = Math.round(0.9 * strip.height);
    let refSum = 0;
    for (let x = 0; x < strip.width; x += 1) refSum += luminance((refY * strip.width + x) * 4);
    alignment = {
      blackKeys: perBar.length,
      sixths: sums.map((sum, i) => (counts[i] > 0 ? sum / counts[i] : null)),
      whiteReference: strip.width > 0 ? refSum / strip.width : null,
    };
  }
  return JSON.stringify({
    keybed: lastValueAfter("keybed"),
    trim: lastValueAfter("trim"),
    detect: lastValueAfter("detect"),
    matcher: lastValueAfter("keys"),
    matched: window.pianocvKeyEvidence ? { dips: window.pianocvKeyEvidence.dips.length, runs: window.pianocvKeyEvidence.runs.length } : null,
    keyRead: read,
    alignment,
    heldQuad: window.pianocvHeldQuad ?? null,
    drawnQuad: window.pianocvFollowedQuad ?? null,
  });
})()`;

async function evaluateClip(cdp, clipUrl, clip) {
  const navStart = Date.now();
  const result = {
    clip,
    heldAtMs: null,
    trimAtMs: null,
    finalTrim: "never measured",
    detectMs: null,
    detectText: "waiting",
    matcher: null,
    matched: null,
    keyRead: null,
    alignment: null,
    heldQuad: null,
    drawnQuad: null,
    steadiness: null,
    error: null,
  };
  try {
    await cdp.send("Page.navigate", { url: clipUrl });
  } catch (error) {
    result.error = error instanceof Error ? error.message : String(error);
    return result;
  }

  let lastConfidence = null;
  let lastConfidenceChangeAt = Date.now();
  let sawRead = false;

  while (Date.now() - navStart < CLIP_BUDGET_MS) {
    const pollStart = Date.now();
    let response;
    try {
      response = await cdp.send("Runtime.evaluate", {
        expression: POLL_EXPRESSION,
        returnByValue: true,
        awaitPromise: true,
      });
    } catch (error) {
      result.error ??= error instanceof Error ? error.message : String(error);
      response = null;
    }
    const exceptionText = response?.result?.exceptionDetails?.text;
    if (exceptionText) result.error ??= exceptionText;
    const raw = response?.result?.result?.value;
    if (typeof raw === "string") {
      let parsed = null;
      try {
        parsed = JSON.parse(raw);
      } catch {
        parsed = null;
      }
      if (parsed) {
        const now = Date.now();
        if (
          result.heldAtMs === null &&
          typeof parsed.keybed === "string" &&
          parsed.keybed.startsWith("held")
        ) {
          result.heldAtMs = now - navStart;
        }
        if (typeof parsed.trim === "string") {
          result.finalTrim = parsed.trim;
          if (result.trimAtMs === null && parsed.trim.includes("%")) {
            result.trimAtMs = now - navStart;
          }
        }
        if (parsed.matched) {
          result.matched = parsed.matched;
        }
        if (typeof parsed.matcher === "string") {
          result.matcher = parsed.matcher;
        }
        if (typeof parsed.detect === "string") {
          result.detectText = parsed.detect;
        }
        if (parsed.keyRead) {
          result.keyRead = parsed.keyRead;
          sawRead = true;
          const confidence =
            typeof parsed.keyRead.confidence === "number"
              ? parsed.keyRead.confidence
              : null;
          if (confidence !== lastConfidence) {
            lastConfidence = confidence;
            lastConfidenceChangeAt = now;
          }
        }
        if (parsed.alignment) result.alignment = parsed.alignment;
        if (parsed.heldQuad) result.heldQuad = parsed.heldQuad;
        if (parsed.drawnQuad) result.drawnQuad = parsed.drawnQuad;
      }
    }
    if (sawRead && Date.now() - lastConfidenceChangeAt >= STABLE_MS) break;
    const elapsed = Date.now() - pollStart;
    if (elapsed < POLL_MS) {
      await new Promise((resolve) => setTimeout(resolve, POLL_MS - elapsed));
    }
  }

  result.steadiness = await cdp
    .send(
      "Runtime.evaluate",
      {
        expression: STEADINESS_EXPRESSION,
        returnByValue: true,
        awaitPromise: true,
      },
      STEADINESS_TIMEOUT_MS,
    )
    .then((response) => response?.result?.result?.value ?? null)
    .catch(() => null);
  if (labelsDir) {
    result.labels = await exportLabels(cdp, clip);
    console.log(`${clip}: ${result.labels}`);
  }
  const detectMatch = /^([0-9.]+)\s*ms$/.exec(result.detectText);
  result.detectMs = detectMatch ? Number(detectMatch[1]) : null;
  const strip = await cdp
    .send("Runtime.evaluate", {
      expression: STRIP_EXPRESSION,
      returnByValue: true,
    })
    .then((response) => response?.result?.result?.value ?? null);
  if (typeof strip === "string") {
    const stripDir = join(root, "data", "evaluations", run);
    await mkdir(stripDir, { recursive: true });
    await writeFile(
      join(stripDir, clip.replace(/\.webm$/, ".png")),
      Buffer.from(strip.split(",")[1], "base64"),
    );
  }
  return result;
}

// PIANOCV_EXPORT_LABELS=<dir> also saves frames the page labelled itself, for training
const labelsDir = process.env.PIANOCV_EXPORT_LABELS
  ? join(root, process.env.PIANOCV_EXPORT_LABELS)
  : null;
const LABELS_EXPRESSION = `import("/src/keylabels.ts").then((m) => m.captureKeyLabels())`;
const LABELS_TIMEOUT_MS = 90_000;

async function exportLabels(cdp, clip) {
  const found = await cdp
    .send(
      "Runtime.evaluate",
      {
        expression: LABELS_EXPRESSION,
        returnByValue: true,
        awaitPromise: true,
      },
      LABELS_TIMEOUT_MS,
    )
    .then((response) => response?.result?.result?.value ?? null)
    .catch(() => null);
  if (found?.kind !== "labelled") return found?.reason ?? "no labels";
  await mkdir(labelsDir, { recursive: true });
  const stem = clip.replace(/\.webm$/, "");
  for (const [i, label] of found.labels.entries()) {
    const name = `${stem}-${String(i).padStart(2, "0")}`;
    await writeFile(
      join(labelsDir, `${name}.png`),
      Buffer.from(label.png.split(",")[1], "base64"),
    );
    await writeFile(
      join(labelsDir, `${name}.ignore.png`),
      Buffer.from(label.ignorePng.split(",")[1], "base64"),
    );
    await writeFile(
      join(labelsDir, `${name}.json`),
      JSON.stringify({ ...label.sidecar, ignoreMask: `${name}.ignore.png` }),
    );
  }
  return `${found.labels.length} labelled`;
}

// how still the drawn keys stay and how well they sit on the picture, measured on the page
const STEADINESS_EXPRESSION = `import("/src/steadiness.ts").then((m) => m.measureSteadiness(3000, 0))`;
const STEADINESS_TIMEOUT_MS = 40_000;

const SYNTHETIC_FRAMES = 240;
const SYNTHETIC_TIMEOUT_MS = 600_000;

// runs the matcher, the fit and the 3D drawing on synthetic renders from their true keybed
// corners, and measures how far each drawn black-key top lands from the rendered one
const BLACK_KEYS_EXPRESSION = `(async () => {
  const ks = await import("/src/keystrip.ts");
  const km = await import("/src/keymatch.ts");
  const { viteAssets } = await import("/src/viteassets.ts");
  const matcher = await km.createKeyMatcher(viteAssets);
  const seg = await import("/src/keyseg.ts");
  const segmenter = await seg.createKeySegmenter(viteAssets).catch(() => null);
  const names = (await fetch("/lab/list/synth-keys").then((r) => r.json())).sort();
  const hull = (points) => {
    const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
    const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    const half = (list) => {
      const out = [];
      for (const p of list) {
        while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], p) <= 0) out.pop();
        out.push(p);
      }
      return out.slice(0, -1);
    };
    return [...half(sorted), ...half([...sorted].reverse())];
  };
  const silhouette = (key) => {
    if (!key.front) return key.top;
    const same = (a, b) => Math.abs(a.x - b.x) < 1e-9 && Math.abs(a.y - b.y) < 1e-9;
    const shared = key.front.filter((c) => key.top.some((t) => same(c, t)));
    const lower = key.front.filter((c) => !key.top.some((t) => same(c, t)));
    if (shared.length !== 2 || lower.length !== 2) return key.top;
    const dx = (lower[0].x + lower[1].x - shared[0].x - shared[1].x) / 2;
    const dy = (lower[0].y + lower[1].y - shared[0].y - shared[1].y) / 2;
    return hull([...key.top, ...key.top.map((p) => ({ x: p.x + dx, y: p.y + dy }))]);
  };
  const scratch = document.createElement("canvas");
  const scratchCtx = scratch.getContext("2d", { willReadFrequently: true });
  // the overlap of two outlines over the pixels of their joint box, the second one minus holes
  const iou = (truthPoly, holes, drawnPoly, size) => {
    const all = [...truthPoly, ...drawnPoly];
    const x0 = Math.floor(Math.min(...all.map((p) => p.x)) * size.width) - 1;
    const y0 = Math.floor(Math.min(...all.map((p) => p.y)) * size.height) - 1;
    const x1 = Math.ceil(Math.max(...all.map((p) => p.x)) * size.width) + 1;
    const y1 = Math.ceil(Math.max(...all.map((p) => p.y)) * size.height) + 1;
    const w = Math.max(1, x1 - x0);
    const h = Math.max(1, y1 - y0);
    if (w * h > 200000) return 0;
    scratch.width = w;
    scratch.height = h;
    const fill = (poly, colour) => {
      scratchCtx.fillStyle = colour;
      scratchCtx.beginPath();
      poly.forEach((p, i) => {
        const x = p.x * size.width - x0;
        const y = p.y * size.height - y0;
        if (i === 0) scratchCtx.moveTo(x, y);
        else scratchCtx.lineTo(x, y);
      });
      scratchCtx.closePath();
      scratchCtx.fill();
    };
    scratchCtx.globalCompositeOperation = "source-over";
    fill(truthPoly, "rgb(255,0,0)");
    for (const hole of holes) fill(hole, "rgb(0,0,0)");
    scratchCtx.globalCompositeOperation = "lighter";
    fill(drawnPoly, "rgb(0,255,0)");
    const data = scratchCtx.getImageData(0, 0, w, h).data;
    let inter = 0;
    let union = 0;
    for (let i = 0; i < data.length; i += 4) {
      const a = data[i] > 127;
      const b = data[i + 1] > 127;
      if (a && b) inter += 1;
      if (a || b) union += 1;
    }
    return union === 0 ? 0 : inter / union;
  };
  const centroid = (poly) => ({
    x: poly.reduce((s, p) => s + p.x, 0) / poly.length,
    y: poly.reduce((s, p) => s + p.y, 0) / poly.length,
  });
  const inside = (p) => p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1;
  const frames = [];
  for (const name of names) {
    if (frames.length >= ${SYNTHETIC_FRAMES}) break;
    const sidecar = await fetch("/lab/data/synth-keys/" + name).then((r) => r.json());
    if (!Array.isArray(sidecar.keys) || !sidecar.corners.every(inside)) continue;
    const image = new Image();
    image.src = "/lab/data/synth-keys/" + name.replace(/\\.json$/, ".png");
    await image.decode();
    const size = { width: image.naturalWidth, height: image.naturalHeight };
    const canvas = document.createElement("canvas");
    canvas.width = size.width;
    canvas.height = size.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0);
    const source = ctx.getImageData(0, 0, size.width, size.height);
    const quad = sidecar.corners;
    const keyWidthPx = 1200 / sidecar.board.whiteKeys;
    // the mask's usual miss on steep views: the far corner on the case side slid along the keys
    const skew = (((frames.length * 7919) % 31) / 15 - 1) * 1.5 * keyWidthPx;
    // every corner off at once, as a mask that is wrong everywhere would be
    const jitter = (k, scale) =>
      ((((frames.length + 1) * (7919 + 104729 * k)) % 97) / 48 - 1) * scale;
    const outlines = {
      true: quad,
      moved: ks.outlineQuad(quad, [
        { x: jitter(0, 1.5 * keyWidthPx), y: jitter(1, 22) },
        { x: 1200 + jitter(2, 1.5 * keyWidthPx), y: jitter(3, 22) },
        { x: 1200 + jitter(4, 1.5 * keyWidthPx), y: 150 + jitter(5, 12) },
        { x: jitter(6, 1.5 * keyWidthPx), y: 150 + jitter(7, 12) },
      ]),
      skewed: ks.outlineQuad(quad, [
        { x: 0, y: 0 },
        { x: 1200 + skew, y: 0 },
        { x: 1200, y: 150 },
        { x: 0, y: 150 },
      ]),
    };
    const width = (key) =>
      Math.hypot(
        (key.top[0].x - key.top[1].x) * size.width,
        (key.top[0].y - key.top[1].y) * size.height,
      );
    const truth = (black) =>
      sidecar.keys
        .filter((key) => key.black === black && key.top.every(inside))
        .map((key) => ({
          c: centroid(key.top),
          w: black ? width(key) / sidecar.keyGeometry.blackWidthFrac : width(key),
        }));
    const errors = (faces, black) => {
      const drawn = [];
      for (let i = 0; i < faces.length; i += 1) {
        if (faces[i].black === black) drawn.push(centroid(faces[i].bar));
        if (faces[i].black) i += 1;
      }
      return truth(black).map(({ c, w }) => {
        let best = Infinity;
        for (const t of drawn) {
          best = Math.min(best, Math.hypot((t.x - c.x) * size.width, (t.y - c.y) * size.height) / w);
        }
        return best;
      });
    };
    const frame = {
      name,
      elevation: sidecar.pose.elevation,
      azimuth: sidecar.pose.azimuth,
      outlines: {},
      overlap: {},
    };
    const blackOutlines = sidecar.keys.filter((key) => key.black).map(silhouette);
    const truthShapes = sidecar.keys
      .filter((key) => key.top.every(inside))
      .map((key) => ({
        black: key.black,
        poly: key.black ? silhouette(key) : key.top,
        c: centroid(key.top),
      }));
    // each true key against the drawn key of its colour whose centre is nearest its own
    const overlaps = (drawn) =>
      ["black", "white"].map((colour) => {
        const black = colour === "black";
        const candidates = drawn.filter((d) => d.black === black);
        return truthShapes
          .filter((t) => t.black === black)
          .map((t) => {
            let best = null;
            let bestDistance = Infinity;
            for (const d of candidates) {
              const c = centroid(d.poly);
              const distance = Math.hypot(c.x - t.c.x, c.y - t.c.y);
              if (distance < bestDistance) {
                bestDistance = distance;
                best = d;
              }
            }
            return best === null ? 0 : iou(t.poly, black ? [] : blackOutlines, best.poly, size);
          });
      });
    if (segmenter) {
      outlines["outlined true"] = (await segmenter.segment(source, size, quad)).outline ?? quad;
      outlines["outlined moved"] =
        (await segmenter.segment(source, size, outlines.moved)).outline ?? outlines.moved;
    }
    for (const [name, given] of Object.entries(outlines)) {
      const matched = await matcher.match(source, given);
      const readQuad = matched?.quad ?? given;
      const strip = ks.rectifyStrip(source, readQuad);
      const found = strip && ks.detectKeys(strip, ks.liftFor(readQuad, size), matched?.evidence ?? null);
      if (found?.kind !== "read") {
        frame.outlines[name] = null;
        continue;
      }
      const read = { ...found, outline: matched?.outline ?? found.outline };
      const faces = ks.projectKeyFaces(read, given, size);
      frame.outlines[name] = {
        raiseMm: read.blackRaiseMm,
        black: errors(faces, true),
        white: errors(faces, false),
      };
      if (name !== "skewed") {
        const fitted = [];
        for (let i = 0; i < faces.length; i += 1) {
          if (faces[i].black) {
            fitted.push({ black: true, poly: hull([...faces[i].bar, ...(faces[i + 1]?.bar ?? [])]) });
            i += 1;
          } else {
            fitted.push({ black: false, poly: faces[i].bar });
          }
        }
        const [black, white] = overlaps(fitted);
        frame.overlap["fitted " + name] = { black, white };
      }
    }
    frames.push(frame);
  }
  return JSON.stringify(frames);
})()`;

function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  return sorted.length === 0 ? null : sorted[Math.floor(sorted.length / 2)];
}

function renderSynthetic(frames) {
  if (!Array.isArray(frames)) return `synthetic keys failed: ${frames?.error}`;
  const buckets = [
    ["all", () => true],
    ["elev<35", (f) => f.elevation < 35],
    ["elev35-55", (f) => f.elevation >= 35 && f.elevation < 55],
    ["elev>=55", (f) => f.elevation >= 55],
  ];
  const cell = (values) => {
    const m = median(values);
    return (m === null ? "-" : m.toFixed(2)).padEnd(10);
  };
  const rows = [
    `synthetic keys: median distance from drawn key to true key, in white-key widths (${frames.length} renders)`,
    ["bucket", "outline", "read", "black", "white", "raise"]
      .map((h) => h.padEnd(10))
      .join(""),
  ];
  for (const [label, keep] of buckets) {
    for (const outline of [
      "true",
      "skewed",
      "moved",
      "outlined true",
      "outlined moved",
    ]) {
      const chosen = frames.filter(keep);
      const read = chosen
        .map((f) => f.outlines[outline])
        .filter((o) => o !== null);
      rows.push(
        [
          label.padEnd(10),
          outline.padEnd(10),
          `${read.length}/${chosen.length}`.padEnd(10),
          cell(read.flatMap((o) => o.black)),
          cell(read.flatMap((o) => o.white)),
          cell(read.map((o) => o.raiseMm)),
        ].join(""),
      );
    }
  }
  const methods = [
    "fitted true",
    "fitted moved",
    "fitted outlined true",
    "fitted outlined moved",
  ];
  rows.push(
    "",
    "synthetic key shapes: mean overlap (IoU) of each true key with its drawn key, 1 is exact",
    ["bucket", ...methods].map((h) => h.padEnd(17)).join(""),
  );
  for (const [label, keep] of buckets) {
    const chosen = frames.filter(keep);
    const mean = (values) =>
      values.length === 0
        ? "-"
        : (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2);
    rows.push(
      [
        label.padEnd(17),
        ...methods.map((m) => {
          const o = chosen.map((f) => f.overlap?.[m]).filter(Boolean);
          return `b ${mean(o.flatMap((x) => x.black))} w ${mean(o.flatMap((x) => x.white))}`.padEnd(
            17,
          );
        }),
      ].join(""),
    );
  }
  return rows.join("\n");
}

// the last key strip with the matcher's edges on it: white-key gaps green, black keys red
const STRIP_EXPRESSION = `(() => {
  const strip = window.pianocvKeyStrip;
  if (!strip) return null;
  const canvas = document.createElement("canvas");
  canvas.width = strip.width;
  canvas.height = strip.height;
  const ctx = canvas.getContext("2d");
  ctx.putImageData(new ImageData(new Uint8ClampedArray(strip.data), strip.width, strip.height), 0, 0);
  const evidence = window.pianocvKeyEvidence;
  if (evidence) {
    ctx.lineWidth = 1;
    ctx.strokeStyle = "lime";
    for (const x of evidence.dips) {
      ctx.beginPath();
      ctx.moveTo(x, strip.height * 0.6);
      ctx.lineTo(x, strip.height);
      ctx.stroke();
    }
    ctx.strokeStyle = "red";
    for (const run of evidence.runs) {
      ctx.strokeRect(run.start, 2, run.end - run.start, strip.height * 0.5);
    }
  }
  return canvas.toDataURL("image/png");
})()`;

function fmt(value, width) {
  const text = value === null || value === undefined ? "-" : String(value);
  return text.length >= width
    ? `${text.slice(0, width - 1)}…`
    : text.padEnd(width);
}

function renderTable(results) {
  const columns = [
    ["clip", 28],
    ["status", 16],
    ["held_ms", 8],
    ["trim_ms", 8],
    ["trim", 14],
    ["keys", 5],
    ["phase", 6],
    ["conf%", 6],
    ["detect_ms", 10],
    ["align_6ths", 24],
    ["white_ref", 9],
    ["black_dark", 11],
    ["white_lit", 10],
    ["swing_px", 10],
  ];
  const header = columns.map(([name, width]) => fmt(name, width)).join(" ");
  const rows = results.map((r) => {
    const status =
      r.error !== null
        ? "error"
        : r.heldAtMs === null
          ? "never held"
          : r.keyRead === null
            ? "held only"
            : r.keyRead.kind === "unsure"
              ? "unsure"
              : "ok";
    const keys = r.keyRead?.kind === "read" ? r.keyRead.whiteKeys : null;
    const phase = r.keyRead?.kind === "read" ? r.keyRead.phase : null;
    const confidence =
      typeof r.keyRead?.confidence === "number"
        ? Math.round(r.keyRead.confidence * 100)
        : null;
    const sixths = r.alignment
      ? r.alignment.sixths
          .map((v) => (v === null ? "-" : Math.round(v)))
          .join("/")
      : null;
    const whiteRef = r.alignment?.whiteReference ?? null;
    const fit = r.steadiness?.fit?.drawn ?? null;
    const pair = (a, b) => (a === null || b === null ? null : `${a}/${b}`);
    return [
      fmt(r.clip, 28),
      fmt(status, 16),
      fmt(r.heldAtMs, 8),
      fmt(r.trimAtMs, 8),
      fmt(r.finalTrim, 14),
      fmt(keys, 5),
      fmt(phase, 6),
      fmt(confidence, 6),
      fmt(r.detectMs, 10),
      fmt(sixths, 24),
      fmt(whiteRef === null ? null : Math.round(whiteRef), 9),
      fmt(fit && pair(fit.blackDarkMedian, fit.blackDarkP10), 11),
      fmt(fit?.whiteBright ?? null, 10),
      fmt(
        r.steadiness &&
          pair(r.steadiness.swingPxMedian, r.steadiness.swingPxP95),
        10,
      ),
    ].join(" ");
  });
  return [header, ...rows].join("\n");
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

const profile = await mkdtemp(join(tmpdir(), "pianocv-keys-eval-chrome-"));
let headless;
let cdp;
try {
  const clips = await clipsToRun();
  const browser = browserPath();
  headless = spawn(
    browser,
    [
      "--headless=new",
      // the hand tracker behind a label's hand masks needs the GPU
      ...(labelsDir ? [] : ["--disable-gpu"]),
      "--no-sandbox",
      "--remote-debugging-port=0",
      "--autoplay-policy=no-user-gesture-required",
      `--user-data-dir=${profile}`,
      "about:blank",
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

  const results = [];
  for (const clip of clips) {
    const clipUrl = `http://127.0.0.1:${vitePort}/?clip=${encodeURIComponent(clip)}`;
    results.push(await evaluateClip(cdp, clipUrl, clip));
  }

  const synthetic =
    process.env.PIANOCV_SYNTHETIC === "none" ? null : await runSynthetic(cdp);

  const reportPath = join(root, "data", "evaluations", `${run}.json`);
  await mkdir(join(root, "data", "evaluations"), { recursive: true });
  await writeFile(
    reportPath,
    JSON.stringify({ run, results, synthetic }, null, 2),
  );

  console.log(renderTable(results));
  if (synthetic !== null) console.log(`\n${renderSynthetic(synthetic)}`);
  console.log(`\nfull report: ${reportPath}`);
} finally {
  if (cdp) cdp.close();
  if (headless) await stop(headless);
  await rm(profile, { recursive: true, force: true });
  await server.close();
}

async function runSynthetic(cdp) {
  await cdp.send("Page.navigate", { url: `http://127.0.0.1:${vitePort}/` });
  await new Promise((resolve) => setTimeout(resolve, 2_000));
  return cdp
    .send(
      "Runtime.evaluate",
      {
        expression: BLACK_KEYS_EXPRESSION,
        returnByValue: true,
        awaitPromise: true,
      },
      SYNTHETIC_TIMEOUT_MS,
    )
    .then((response) => {
      const value = response?.result?.result?.value;
      return typeof value === "string"
        ? JSON.parse(value)
        : { error: response?.result?.exceptionDetails?.exception?.description };
    });
}
