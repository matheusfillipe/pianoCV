BUN := bun --cwd=web
UV := uv run --project tools
MODEL_REPO := mattf/pianoCV
KAGGLE_SEG2_OUTPUT_DIR ?= data/models/kaggle-keybed-seg2-real
PORT ?= 5274

.DEFAULT_GOAL := help
.PHONY: help install fix precommit check list-lab-data lab-extract lab-train lab-detect \
        lab-seg2-kaggle-run lab-seg2-kaggle-output lab-browser-gridtest lab-keys-eval \
        lab-synth-generate lab-synth-motion \
        tools-fix tools-format-check tools-lint tools-typecheck \
        tools-test tools-coverage tools-dead-code tools-unused-deps tools-security tools-audit tools-upgrade \
        build web-typecheck web-lint web-fix web-test web-build dev dev-alt model site publish-models clean lab-export \
        lab-evaluate lab-real-seg2-prepare lab-relabel-keys lab-compare lab-trainseg2 lab-dataset-push \
        lab-keymatch-train lab-keymatch-push lab-keyseg-train lab-keyseg-labels lab-keyseg-push lab-keynet-push lab-keynet-train lab-keynet-eval lab-synth-test \
        core-lint core-test core-wasm core-fix lab-mlflow-log

help: ## list available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## install dependencies (bun, uv) and git hooks
	$(BUN) install
	uv sync --project tools
	$(UV) pre-commit install
	@# the agent plugin is local tooling, absent from a fresh clone
	@[ -f .opencode/package.json ] && (cd .opencode && bun install) || true

fix: tools-fix web-fix ## autofix formatting and lint issues (ruff, biome)

precommit: fix ## hook entry: same as fix

# --- checks (verify, never produce artifacts) ---

check: tools-format-check tools-lint tools-typecheck tools-test core-lint core-test web-typecheck web-types web-lint web-test web-build ## run all checks (the pre-commit gate)

quality: check tools-dead-code tools-unused-deps tools-security tools-audit tools-coverage build ## run the full quality gate
	@echo "quality gate passed"

tools-fix: ## autofix python formatting and lint (ruff)
	cd tools && uv run ruff format src tests && uv run ruff check --fix src tests

tools-format-check: ## check python formatting (ruff format)
	cd tools && uv run ruff format --check src tests

tools-lint: ## lint python (ruff)
	cd tools && uv run ruff check src tests

tools-typecheck: ## typecheck python (mypy strict)
	cd tools && uv run mypy

tools-test: ## run python tests (pytest)
	cd tools && uv run pytest

tools-coverage: ## run python tests with coverage (pytest-cov)
	cd tools && uv run pytest --cov --cov-report=term-missing

tools-dead-code: ## detect dead python code (vulture)
	cd tools && uv run vulture src/pianocv tests

tools-unused-deps: ## detect unused python dependencies (deptry)
	cd tools && uv run deptry .

tools-security: ## scan python for security issues (bandit)
	cd tools && uv run bandit -c pyproject.toml -r src/pianocv

tools-audit: ## audit python dependencies for vulnerabilities (pip-audit)
	cd tools && uv run --with pip pip-audit

tools-upgrade: ## move the named python packages to their newest allowed versions: make tools-upgrade PKGS="urllib3 virtualenv"
	cd tools && uv lock $(addprefix --upgrade-package ,$(PKGS)) && uv sync

core-lint: ## check rust formatting and lint the keycore crate (rustfmt, clippy)
	cd core && cargo fmt --check && cargo clippy --all-targets -- -D warnings

core-fix: ## format the keycore crate (rustfmt)
	cd core && cargo fmt

core-test: ## run the keycore crate tests (cargo test)
	cd core && cargo test

core-wasm: ## build the keycore crate for the browser into web/src/keycore-wasm (wasm-pack)
	cd core && wasm-pack build --target web --release --out-dir ../web/src/keycore-wasm --out-name keycore

build: ## build the python package (uv build)
	uv build --project tools

web-typecheck: core-wasm ## typecheck web (tsc)
	$(BUN) run typecheck

web-types: ## write the published type declarations (tsc)
	$(BUN) run types

web-lint: ## lint web (biome check)
	$(BUN) run lint

web-fix: ## autofix web formatting and lint (biome)
	$(BUN) run fix

web-test: core-wasm ## run web tests (vitest)
	$(BUN) run test

web-build: ## bundle the web app (vite build)
	$(BUN) run build

dev: ## run the web dev server (vite)
	$(BUN) run dev

dev-alt: ## run the web dev server on another port, for when 5273 is already held by a different checkout (make dev-alt PORT=5274)
	$(BUN) run dev -- --port $(PORT)

MODELS := keybed_seg2.onnx keymatch.onnx keyseg.onnx

model: ## download the trained models from hugging face into web/public
	@for model in $(MODELS); do \
		curl -fL --create-dirs -o web/public/$$model \
			https://huggingface.co/$(MODEL_REPO)/resolve/main/$$model || exit 1; \
	done
	@ls -lh $(addprefix web/public/,$(MODELS))

MESSAGE ?= update the models

publish-models: ## check web/public's models for leaked local paths, then upload them and hf-model/README.md to hugging face (needs hf auth login; MESSAGE="...")
	@for model in $(MODELS); do \
		if strings -n 6 web/public/$$model | grep -qE '/Users/|/home/|site-packages|stack_trace'; then \
			echo "$$model carries a local path or stack trace, not publishing it"; exit 1; \
		fi; \
	done
	@rm -rf .publish && mkdir .publish
	cp $(addprefix web/public/,$(MODELS)) hf-model/README.md .publish/
	hf upload $(MODEL_REPO) .publish . --commit-message "$(MESSAGE)"
	@rm -rf .publish

BASE ?= /

site: ## build the static site with its models into web/dist, served under BASE (make site BASE=/pianoCV/)
	$(BUN) install --frozen-lockfile
	$(MAKE) model
	PIANOCV_BASE=$(BASE) $(BUN) run build

list-lab-data: ## list saved lab recordings (data/recordings)
	mkdir -p data/recordings && ls -la data/recordings

lab-extract: ## extract labeled frames from recordings (data/recordings -> data/frames)
	cd tools && uv run python -m pianocv.dataset

lab-train: ## train detector on synthetic renders then fine-tune on real rec frames (data/models/keybed_net.pt)
	cd tools && uv run python -m pianocv.train

lab-export: ## export the trained detector to web/public/keybed_net.onnx
	cd tools && uv run python -m pianocv.export

lab-train-seg: ## train the segmentation detector on renders plus data/synth (data/models/keybed_seg.pt)
	cd tools && uv run python -m pianocv.trainseg $(ARGS)

lab-export-seg: ## export the trained segmentation detector to web/public/keybed_seg.onnx
	cd tools && uv run python -m pianocv.export --seg

lab-dump: ## render procedural frames to disk (data/procedural)
	cd tools && uv run python -m pianocv.dump $(ARGS)

lab-synth-generate: ## render synthetic keyboards with a randomised case into data/synth-case, unattended (PIANOCV_GEN_FRAMES, default 8000)
	cd web && bun gen-runner.mjs

lab-synth-motion: ## render moving-camera sequences with motion blur into data/synth-motion, unattended (PIANOCV_GEN_FRAMES, default 8000)
	cd web && PIANOCV_GEN_MOTION=1 PIANOCV_GEN_OUT=synth-motion bun gen-runner.mjs

lab-synth-test: ## render a held-out set of moving-camera frames with exact labels into data/synth-test, never trained on (PIANOCV_GEN_FRAMES, default 400)
	cd web && PIANOCV_GEN_MOTION=1 PIANOCV_GEN_OUT=synth-test PIANOCV_GEN_FRAMES=$${PIANOCV_GEN_FRAMES:-400} bun gen-runner.mjs

lab-export-seg2: ## export the pretrained segmentation detector to web/public/keybed_seg2.onnx
	cd tools && uv run python -m pianocv.export --seg2 $(ARGS)

lab-jitter: ## measure how much the detection moves on static recordings
	cd tools && uv run python -m pianocv.jitter $(ARGS)

lab-gridtest: ## score the detector per pose on the deterministic render grid (data/grid)
	cd tools && uv run python -m pianocv.gridtest $(ARGS)

lab-browser-gridtest: ## run the served browser ONNX model over the deterministic 3D grid
	cd web && bun grid-eval-runner.mjs

lab-keys-eval: ## run the live keyboard pipeline headless on every saved recording and report hold time, trim, key count and black-key alignment
	cd web && bun keys-eval-runner.mjs

lab-keyseg-labels: ## label frames of every saved recording with the keys the app fits to them, for training (data/real-keys)
	cd web && PIANOCV_EXPORT_LABELS=data/real-keys PIANOCV_SYNTHETIC=none bun keys-eval-runner.mjs

lab-corpus-bake: ## bake data/synth down to the net's input size (data/corpus); ARGS="--synth-dir ... --out-dir ..." to bake other directories, repeat --synth-dir to merge several
	cd tools && uv run python -m pianocv.bake $(ARGS)

lab-corpus-zip: lab-corpus-bake ## pack data/corpus into data/keybed-corpus.zip for upload to a training host
	rm -f data/keybed-corpus.zip
	cd data && zip -q -r keybed-corpus.zip corpus -x '.*'
	@ls -lh data/keybed-corpus.zip

lab-detect: lab-extract ## evaluate keybed detector on extracted frames, split by fine-tuned vs held out
	cd tools && uv run python -m pianocv.evaluate $(ARGS)

lab-evaluate: ## evaluate a detector on already-extracted labelled frames
	cd tools && uv run python -m pianocv.evaluate $(ARGS)

lab-real-seg2-prepare: ## prepare real labelled frames for SegNet2 fine-tuning
	cd tools && uv run python -m pianocv.realseg2 $(ARGS)

lab-relabel-keys: ## relabel the real frames and each recording's truth keys-only, with the live far-edge trim (data/real-seg2-keys, data/recordings-keys-truth.json)
	cd web && bun relabel-keys-runner.mjs

lab-compare: ## compare onnx keybed detectors on the render grid and on real recordings
	$(UV) python -m pianocv.compare $(ARGS)

lab-trainseg2: ## fine-tune KeybedSegNet2 on data/corpus and data/real-seg2 (data/models/seg2-tuned)
	cd tools && uv run python -m pianocv.trainseg2 $(ARGS)

lab-dataset-push: ## pack data/corpus, data/real-seg2 and pianocv into a bundle and upload it to an S3 bucket with mc (ARGS="--alias <mc alias> --version <v>")
	cd tools && uv run python -m pianocv.datasetpush $(ARGS)

lab-keymatch-train: ## train the key matcher locally on data/synth-keys
	cd tools && uv run python -m pianocv.trainkeymatch $(ARGS)

lab-keyseg-train: ## train the per-key segmenter locally on data/synth-keys; ARGS="--preview 12" draws labels only
	cd tools && uv run python -m pianocv.trainkeyseg $(ARGS)

lab-keyseg-push: ## pack data/real-keys, pianocv and the published keyseg.onnx for a fine-tune and upload them with mc (ARGS="--alias <mc alias> --version <v>")
	cd tools && uv run python -m pianocv.keysegpush $(ARGS)

lab-keynet-push: ## pack data/real-keys, data/synth-motion, pianocv and keyseg.onnx for a KeyNet run and upload them with mc (ARGS="--alias <mc alias> --version <v>")
	cd tools && uv run python -m pianocv.keysegpush --name keynet --motion-dir ../data/synth-motion $(ARGS)

lab-mlflow-log: ## log a model's scores from a score JSON onto its MLflow run (ARGS="--tracking-uri <url> --run-id <id> --scores <json> --model <file> --prefix <what>")
	cd tools && uv run python -m pianocv.mlflowlog $(ARGS)

lab-keynet-train: ## train KeyNet locally; ARGS="--still-dir ../data/synth-keys ..." (see --help)
	cd tools && uv run python -m pianocv.trainkeynet $(ARGS)

lab-keynet-eval: ## score KeyNet exports on held-out labelled frames, in key widths (ARGS="--model <onnx> --clips <recs> [--decoder parabola]")
	cd tools && uv run python -m pianocv.evalkeynet --frames ../data/real-keys $(ARGS)

lab-keymatch-push: ## pack data/synth-keys and pianocv into a bundle and upload it to an S3 bucket with mc (ARGS="--alias <mc alias> --version <v>")
	cd tools && uv run python -m pianocv.keymatchpush $(ARGS)

lab-seg2-kaggle-run: ## push and start the private real-frame Seg2 fine-tuning notebook
	kaggle kernels push -p tools/kaggle/keybed-seg2

lab-seg2-kaggle-output: ## download completed private real-frame Seg2 artifacts
	mkdir -p $(KAGGLE_SEG2_OUTPUT_DIR)
	kaggle kernels output mattflyx/keybed-segmentation-seg2 -p $(KAGGLE_SEG2_OUTPUT_DIR) --force

clean: ## remove local caches and build artifacts
	rm -rf tools/.ruff_cache tools/.mypy_cache tools/.pytest_cache tools/.coverage tools/.coverage.* tools/.vulture web/dist
