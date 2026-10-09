BUN := bun --cwd=web
UV := uv run --project tools
MODEL_REPO := mattf/pianoCV
PORT ?= 5274

.DEFAULT_GOAL := help
.PHONY: help install fix precommit check list-lab-data lab-synth-generate lab-synth-motion lab-synth-test \
        tools-fix tools-format-check tools-lint tools-typecheck \
        tools-test tools-coverage tools-dead-code tools-unused-deps tools-security tools-audit tools-upgrade \
        build web-typecheck web-types web-lint web-fix web-test web-build dev dev-alt model site publish-models clean \
        lab-keymatch-push lab-keynet-push lab-keynet-train lab-keynet-eval lab-keynet-fp16 lab-mlflow-log \
        core-lint core-test core-native-test core-backedge core-demo core-wasm core-fix

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

check: tools-format-check tools-lint tools-typecheck tools-test core-lint core-test core-native-test web-typecheck web-types web-lint web-test web-build ## run all checks (the pre-commit gate)

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
	cd core && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo clippy --all-targets --features native -- -D warnings

core-fix: ## format the keycore crate (rustfmt)
	cd core && cargo fmt

core-test: ## run the keycore crate tests (cargo test)
	cd core && cargo test

core-native-test: ## run the keycore crate tests with the native engine (cargo test --features native, needs web/public/keynet.onnx from make model)
	cd core && cargo test --features native

core-backedge: ## score how far the fitted back edge lands from labelled key backs, per case colour (ARGS="<labels-dir> [images-dir] [model]")
	cd core && cargo run --release --features native --example backedge -- $(ARGS)

core-demo: ## draw the keys on a video with the native engine (ARGS="<video> <out-dir>")
	cd core && cargo run --release --features native --example desktop -- $(ARGS)

core-wasm: ## build the keycore crate for the browser into web/src/keycore-wasm (wasm-pack)
	cd core && wasm-pack build --target web --release --out-dir ../web/src/keycore-wasm --out-name keycore
	@# wasm-pack ignores its own output for git, which would also keep it out of the npm package
	rm -f web/src/keycore-wasm/.gitignore

build: ## build the python package (uv build)
	uv build --project tools

web-typecheck: core-wasm ## typecheck web (tsc)
	$(BUN) run typecheck

web-types: core-wasm ## write the published type declarations (tsc)
	$(BUN) run types

web-lint: ## lint web (biome check)
	$(BUN) run lint

web-fix: ## autofix web formatting and lint (biome)
	$(BUN) run fix

web-test: core-wasm ## run web tests (vitest)
	$(BUN) run test

web-build: core-wasm ## bundle the web app (vite build)
	$(BUN) run build

dev: core-wasm ## run the web dev server (vite)
	$(BUN) run dev

dev-alt: core-wasm ## run the web dev server on another port, for when 5273 is already held by a different checkout (make dev-alt PORT=5274)
	$(BUN) run dev -- --port $(PORT)

MODELS := keynet.onnx

model: ## download the trained model from hugging face into web/public
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
	$(MAKE) model core-wasm
	PIANOCV_BASE=$(BASE) $(BUN) run build

list-lab-data: ## list saved lab recordings (data/recordings)
	mkdir -p data/recordings && ls -la data/recordings

lab-synth-generate: ## render synthetic keyboards with a randomised case into data/synth-case, unattended (PIANOCV_GEN_FRAMES, default 8000)
	cd web && bun gen-runner.mjs

lab-synth-motion: ## render moving-camera sequences with motion blur into data/synth-motion, unattended (PIANOCV_GEN_FRAMES, default 8000)
	cd web && PIANOCV_GEN_MOTION=1 PIANOCV_GEN_OUT=synth-motion bun gen-runner.mjs

lab-synth-test: ## render a held-out set of moving-camera frames with exact labels into data/synth-test, never trained on (PIANOCV_GEN_FRAMES, default 400)
	cd web && PIANOCV_GEN_MOTION=1 PIANOCV_GEN_OUT=synth-test PIANOCV_GEN_FRAMES=$${PIANOCV_GEN_FRAMES:-400} bun gen-runner.mjs





lab-keynet-push: ## pack data/real-keys, data/synth-motion, pianocv and keyseg.onnx for a KeyNet run and upload them with mc (ARGS="--alias <mc alias> --version <v>")
	cd tools && uv run python -m pianocv.keysegpush --name keynet --motion-dir ../data/synth-motion $(ARGS)

lab-keynet-fp16: ## convert a KeyNet export to half precision (ARGS="<source.onnx> <target.onnx>")
	cd tools && uv run python -m pianocv.keynetfp16 $(ARGS)

lab-mlflow-log: ## log a model's scores from a score JSON onto its MLflow run (ARGS="--tracking-uri <url> --run-id <id> --scores <json> --model <file> --prefix <what>")
	cd tools && uv run python -m pianocv.mlflowlog $(ARGS)

lab-keynet-train: ## train KeyNet locally; ARGS="--still-dir ../data/synth-keys ..." (see --help)
	cd tools && uv run python -m pianocv.trainkeynet $(ARGS)

lab-keynet-eval: ## score KeyNet exports on held-out labelled frames, in key widths (ARGS="--model <onnx> --clips <recs> [--decoder parabola]")
	cd tools && uv run python -m pianocv.evalkeynet --frames ../data/real-keys $(ARGS)

lab-keymatch-push: ## pack data/synth-keys and pianocv into a bundle and upload it to an S3 bucket with mc (ARGS="--alias <mc alias> --version <v>")
	cd tools && uv run python -m pianocv.keymatchpush $(ARGS)

clean: ## remove local caches and build artifacts
	rm -rf tools/.ruff_cache tools/.mypy_cache tools/.pytest_cache tools/.coverage tools/.coverage.* tools/.vulture web/dist
