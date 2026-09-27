# pianoCV

Spike: automatic video editing for piano video. A camera watches hands on a real keyboard; the browser pipeline detects the keybed, solves its 3D plane, and composites kinesthesia's MIDI-driven visuals into the original camera perspective. CV does geometry only, never notes. The runtime is browser-only (web/). Python (tools/) is the offline lab. Eventual home: kinesthesia.

## Where things live
- web/ TS browser app (Vite + MediaPipe tasks-vision): the runtime
- tools/ python uv package pianocv: offline lab (synthetic data, frame validation, detector training)
- docs/ untracked local research notes
- The Makefile is the single canonical interface for all checks; CI and pre-commit both call it.

## Stack
web: Vite, TypeScript strict, Biome, @mediapipe/tasks-vision. tools: uv, ruff, mypy strict, pytest.

## Commands (Makefile is SSoT)
All commands go through the Makefile. Never call uv/bun/tsc/biome binaries directly; add or extend a make target instead.

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
- Browser-only runtime: nothing in web/ may depend on a native process.
- tools/ is offline lab only and must not become a runtime dependency.
- Camera recordings and generated datasets stay in data/ (gitignored) and never enter git.

## Hygiene
No secrets in the repo. Recordings of people are personal data: local only.
