import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";

const ROUTE =
  /^\/lab\/save\/(?:(synth|synth-case|synth-keys|synth-motion|synth-test|real-keys-fixed|grid|evaluations)\/)?([^/]+)$/;
const LIST_ROUTE =
  /^\/lab\/list\/(grid|synth|synth-case|synth-keys|synth-motion|real-keys|real-keys-fixed)$/;
const DATA_ROUTE =
  /^\/lab\/data\/(grid|synth|synth-case|synth-keys|synth-motion|real-keys|real-keys-fixed|evaluations|models)\/([^/?]+)$/;
const CLIP_ROUTE = /^\/lab\/clip\/([^/?]+)$/;
const DIRS: Record<string, string> = {
  recordings: "recordings",
  synth: "synth",
  "synth-case": "synth-case",
  "synth-keys": "synth-keys",
  "synth-motion": "synth-motion",
  "synth-test": "synth-test",
  "real-keys": "real-keys",
  "real-keys-fixed": "real-keys-fixed",
  models: "models",
  grid: "grid",
  evaluations: "evaluations",
};

function contentType(name: string): string {
  if (name.endsWith(".json")) {
    return "application/json";
  }
  return name.endsWith(".onnx") ? "application/octet-stream" : "image/png";
}
const UNSAFE_NAME = /[^a-zA-Z0-9._-]/g;

export function labServer(): Plugin {
  const dataDir = join(fileURLToPath(new URL("..", import.meta.url)), "data");
  return {
    name: "pianocv-lab-server",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const list = LIST_ROUTE.exec(req.url ?? "");
        if (req.method === "GET" && list) {
          const directory = join(dataDir, DIRS[list[1]]);
          readdir(directory)
            .then((entries) => {
              res.setHeader("content-type", "application/json");
              res.end(
                JSON.stringify(
                  entries.filter((entry) => entry.endsWith(".json")),
                ),
              );
            })
            .catch(() => {
              res.statusCode = 404;
              res.end();
            });
          return;
        }
        const data = DATA_ROUTE.exec(req.url ?? "");
        if (req.method === "GET" && data) {
          const name = data[2].replace(UNSAFE_NAME, "");
          readFile(join(dataDir, DIRS[data[1]], name))
            .then((body) => {
              res.setHeader("content-type", contentType(name));
              res.end(body);
            })
            .catch(() => {
              res.statusCode = 404;
              res.end();
            });
          return;
        }
        const clip = CLIP_ROUTE.exec(req.url ?? "");
        if (req.method === "GET" && clip) {
          const name = clip[1].replace(UNSAFE_NAME, "");
          readFile(join(dataDir, DIRS.recordings, name))
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
        if (req.method !== "POST") {
          next();
          return;
        }
        const match = ROUTE.exec(req.url ?? "");
        if (!match) {
          next();
          return;
        }
        const recordDir = join(dataDir, DIRS[match[1] ?? "recordings"]);
        const name = match[2].replace(UNSAFE_NAME, "");
        if (!name) {
          res.statusCode = 400;
          res.end();
          return;
        }
        const chunks: Buffer[] = [];
        req.on("data", (chunk: Buffer) => {
          chunks.push(chunk);
        });
        req.on("end", () => {
          mkdir(recordDir, { recursive: true })
            .then(() => writeFile(join(recordDir, name), Buffer.concat(chunks)))
            .then(() => {
              res.statusCode = 204;
              res.end();
            })
            .catch(() => {
              res.statusCode = 500;
              res.end();
            });
        });
      });
    },
  };
}
