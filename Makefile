BUN := bun --cwd=web
UV := uv run --project tools
MODEL_REPO := mattf/keybed-seg
KAGGLE_KEY_INSTANCES_DIR ?= data/key-instances/kaggle-20260916-v2
KAGGLE_KEY_INSTANCES_VERSION_MESSAGE ?= include per-key target folders
KAGGLE_KEY_ALIGNMENT_OUTPUT_DIR ?= data/models/kaggle-key-alignment

.DEFAULT_GOAL := help
.PHONY: help install fix precommit check list-lab-data lab-extract lab-train lab-detect \
        lab-key-instances-prepare lab-key-instances-archive \
        lab-key-instances-kaggle-push lab-key-instances-kaggle-version lab-key-instances-kaggle-run \
        lab-key-instances-kaggle-output lab-key-instances-kaggle-evaluate \
        lab-calibration-eval \
        tools-fix tools-lock tools-format-check tools-lint tools-typecheck \
        tools-test tools-coverage tools-dead-code tools-unused-deps tools-security tools-audit \
        build web-typecheck web-lint web-fix web-test web-build dev model clean lab-export lab-evaluate

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

check: tools-format-check tools-lint tools-typecheck tools-test lab-key-instances-kaggle-check web-typecheck web-types web-lint web-test web-build ## run all checks (the pre-commit gate)

quality: check tools-dead-code tools-unused-deps tools-security tools-audit tools-coverage build ## run the full quality gate
	@echo "quality gate passed"

tools-fix: ## autofix python formatting and lint (ruff)
	cd tools && uv run ruff format src tests && uv run ruff check --fix src tests

tools-lock: ## refresh the Python dependency lock after metadata changes
	uv lock --project tools

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
	cd tools && uv run vulture src/kvt tests

tools-unused-deps: ## detect unused python dependencies (deptry)
	cd tools && uv run deptry .

tools-security: ## scan python for security issues (bandit)
	cd tools && uv run bandit -c pyproject.toml -r src/kvt

tools-audit: ## audit python dependencies for vulnerabilities (pip-audit)
	cd tools && uv run --with pip pip-audit

build: ## build the python package (uv build)
	uv build --project tools

web-typecheck: ## typecheck web (tsc)
	$(BUN) run typecheck

web-types: ## write the published type declarations (tsc)
	$(BUN) run types

web-lint: ## lint web (biome check)
	$(BUN) run lint

web-fix: ## autofix web formatting and lint (biome)
	$(BUN) run fix

web-test: ## run web tests (vitest)
	$(BUN) run test

web-build: ## bundle the web app (vite build)
	$(BUN) run build

dev: ## run the web dev server (vite)
	$(BUN) run dev

model: ## download the trained detector from hugging face into web/public
	curl -fL --create-dirs -o web/public/keybed_seg2.onnx \
		https://huggingface.co/$(MODEL_REPO)/resolve/main/keybed_seg2.onnx
	@ls -lh web/public/keybed_seg2.onnx

list-lab-data: ## list saved lab recordings (data/recordings)
	mkdir -p data/recordings && ls -la data/recordings

lab-extract: ## extract labeled frames from recordings (data/recordings -> data/frames)
	cd tools && uv run python -m kvt.dataset

lab-train: ## train detector on synthetic renders then fine-tune on real rec frames (data/models/keybed_net.pt)
	cd tools && uv run python -m kvt.train

lab-export: ## export the trained detector to web/public/keybed_net.onnx
	cd tools && uv run python -m kvt.export

lab-train-seg: ## train the segmentation detector on renders plus data/synth (data/models/keybed_seg.pt)
	cd tools && uv run python -m kvt.trainseg $(ARGS)

lab-export-seg: ## export the trained segmentation detector to web/public/keybed_seg.onnx
	cd tools && uv run python -m kvt.export --seg

lab-dump: ## render procedural frames to disk (data/procedural)
	cd tools && uv run python -m kvt.dump $(ARGS)

lab-export-seg2: ## export the pretrained segmentation detector to web/public/keybed_seg2.onnx
	cd tools && uv run python -m kvt.export --seg2 $(ARGS)

lab-jitter: ## measure how much the detection moves on static recordings
	cd tools && uv run python -m kvt.jitter $(ARGS)

lab-gridtest: ## score the detector per pose on the deterministic render grid (data/grid)
	cd tools && uv run python -m kvt.gridtest $(ARGS)

lab-calibration-eval: ## run a bounded headless browser calibration evaluation (data/evaluations)
	cd web && bun eval-runner.mjs

lab-corpus-bake: ## bake data/synth down to the net's input size (data/corpus)
	cd tools && uv run python -m kvt.bake

lab-corpus-zip: lab-corpus-bake ## pack data/corpus into data/keybed-corpus.zip for upload to a training host
	rm -f data/keybed-corpus.zip
	cd data && zip -q -r keybed-corpus.zip corpus -x '.*'
	@ls -lh data/keybed-corpus.zip

lab-detect: lab-extract ## evaluate keybed detector on extracted frames, split by fine-tuned vs held out
	cd tools && uv run python -m kvt.evaluate $(ARGS)

lab-evaluate: ## evaluate a detector on already-extracted labelled frames
	cd tools && uv run python -m kvt.evaluate $(ARGS)

lab-key-instances-prepare: ## make a separate Kaggle-ready per-key dataset from data/key-instances
	cd tools && uv run python -m kvt.keyinstances $(ARGS)

lab-key-instances-archive: ## build a Kaggle upload zip from the prepared per-key dataset
	cd tools && uv run python -m kvt.keyinstances --archive $(ARGS)

lab-key-instances-kaggle-push: ## publish prepared data to the private Kaggle dataset
	kaggle datasets create -p $(KAGGLE_KEY_INSTANCES_DIR) --dir-mode zip

lab-key-instances-kaggle-version: ## publish a corrected or newer private Kaggle dataset version
	kaggle datasets version -p $(KAGGLE_KEY_INSTANCES_DIR) --dir-mode zip --message "$(KAGGLE_KEY_INSTANCES_VERSION_MESSAGE)"

lab-key-instances-kaggle-run: ## push and start the private GPU alignment-training notebook
	kaggle kernels push -p tools/kaggle/key-instance-alignment

lab-key-instances-kaggle-output: ## download completed private alignment-model artifacts
	mkdir -p $(KAGGLE_KEY_ALIGNMENT_OUTPUT_DIR)
	kaggle kernels output mattflyx/piano-key-instance-alignment -p $(KAGGLE_KEY_ALIGNMENT_OUTPUT_DIR) --force

lab-key-instances-kaggle-evaluate: ## evaluate a downloaded alignment ONNX artifact locally
	cd tools && uv run python -m kvt.keyalignmenteval

lab-key-instances-kaggle-check: ## syntax-check the Kaggle alignment-training script
	$(UV) python -m py_compile tools/kaggle/key-instance-alignment/key-instance-alignment.py

clean: ## remove local caches and build artifacts
	rm -rf tools/.ruff_cache tools/.mypy_cache tools/.pytest_cache tools/.coverage tools/.coverage.* tools/.vulture web/dist
