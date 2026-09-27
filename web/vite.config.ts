import { resolve } from "node:path";
import { defineConfig } from "vite";
import { labServer } from "./lab-server.ts";

export default defineConfig({
  // a project page serves the app under its repo's path, so a deploy build sets it
  base: process.env.PIANOCV_BASE ?? "/",
  plugins: [labServer()],
  server: { port: 5273, strictPort: true },
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
        gen: resolve(import.meta.dirname, "gen.html"),
        hands: resolve(import.meta.dirname, "hands.html"),
        gridEval: resolve(import.meta.dirname, "grid-eval.html"),
      },
    },
  },
});
