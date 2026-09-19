import { resolve } from "node:path";
import { defineConfig } from "vite";
import { labServer } from "./lab-server.ts";

export default defineConfig({
  plugins: [labServer()],
  server: { port: 5273, strictPort: true },
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
        gen: resolve(import.meta.dirname, "gen.html"),
        hands: resolve(import.meta.dirname, "hands.html"),
        eval: resolve(import.meta.dirname, "eval.html"),
        gridEval: resolve(import.meta.dirname, "grid-eval.html"),
      },
    },
  },
});
