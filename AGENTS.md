# pianoCV

Spike: automatic video editing for piano video. A camera watches hands on a real keyboard; KeyNet finds the keys' keypoints, a Rust core fits the keyboard to them, and the app draws kinesthesia's MIDI-driven visuals in the camera's own perspective. CV does geometry only, never notes. The runtime is the browser app (web/) with the Rust core (core/) compiled to WebAssembly. Python (tools/) is the offline lab. Eventual home: kinesthesia.

## Direction
- One portable keypoint model (KeyNet, ONNX) does the perception, so the same model runs real-time AR on mobile, desktop and the web.
- The code around the model is plain geometry only: decode peaks, fit the keyboard template, track, draw. It stays small enough to port to every platform.
- When the drawing is wrong, we fix the model or its labels. We never add computer vision heuristics to the runtime.

## Where things live
- web/ TS browser app (Vite, ONNX Runtime Web, MediaPipe hands): camera, model run and drawing
- core/ Rust crate keycore: the geometry after the model (decode, fit, track, key faces), built to WebAssembly for web and natively for other platforms (cargo feature `native`)
- tools/ python uv package pianocv: offline lab (synthetic data, KeyNet training and scoring)
- docs/ untracked local research notes
- The Makefile is the single canonical interface for all checks; CI and pre-commit both call it.

## Stack
web: Vite, TypeScript strict, Biome, onnxruntime-web, @mediapipe/tasks-vision. core: Rust, clippy, rustfmt, wasm-pack. tools: uv, ruff, mypy strict, pytest.

## Commands (Makefile is SSoT)
All commands go through the Makefile. Never call uv/bun/tsc/biome/cargo/wasm-pack binaries directly; add or extend a make target instead.

## Before considering work complete
1. Run `make quality`.
2. Fix all failures.
3. Do not weaken or remove quality checks to make them pass.
4. Do not leave unused dependencies or dead code.

## Code style
- Python: imports at top of module, `list[str]` and `X | None`, no `Any`, no bare `except`, mypy strict clean.
- TypeScript: strict mode, no `any`, no non-null assertions.
- Comments only to explain the non-obvious why. Default to zero.
- Prose (docs, commits): declarative, terse, no em dashes.

## Conventions
- Nothing in web/ may depend on a native process; core/ is the only code shared across platforms.
- tools/ is offline lab only and must not become a runtime dependency.
- Camera recordings and generated datasets stay in data/ (gitignored) and never enter git.

## Hygiene
No secrets in the repo. Recordings of people are personal data: local only.
