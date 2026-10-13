# JamMate developer commands. Run `make` or `make help` for the list.
# Override variables on the command line, e.g. `make run OUTPUT=speakers`.

INPUT      ?= usb
OUTPUT     ?= speakers # usb
CHANNEL    ?= 1
BLOCK      ?= 128
GAIN       ?= 1.0
INSTRUMENT ?= guitar
STYLE      ?= rock
MODEL      ?= mrt2_small
FILE       ?=
OUT        ?= out.wav
DURATION   ?=
PROMPT     ?= stoner rock

.DEFAULT_GOAL := help
.PHONY: help sync test lint format check devices gui run capture analyze render app mrt-check clean

help: ## Show this help
	@echo "Usage: make <target> [VAR=value ...]"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Variables (current value):"
	@echo "  INPUT=$(INPUT)  OUTPUT=$(OUTPUT)  CHANNEL=$(CHANNEL)  BLOCK=$(BLOCK)  GAIN=$(GAIN)"
	@echo "  INSTRUMENT=$(INSTRUMENT)  STYLE=$(STYLE)  MODEL=$(MODEL)"
	@echo "  FILE=$(FILE)  OUT=$(OUT)  DURATION=$(DURATION)  PROMPT=$(PROMPT)"

sync: ## Install / update the environment (uv sync)
	uv sync

test: ## Run the unit tests (no hardware needed)
	uv run pytest

lint: ## Lint with ruff
	uv run ruff check .

format: ## Auto-format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

check: ## Lint, format check and tests (run before finishing any change)
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest

devices: ## List audio devices and what INPUT/OUTPUT select
	uv run jammate --list-devices --input "$(INPUT)" --output "$(OUTPUT)"

gui: ## Launch the PySide6 desktop GUI (instrument, model & device refresh built in)
	uv run jammate --instrument $(INSTRUMENT) --model-size $(MODEL)

run: ## Headless runner: live notes + status line (e.g. OUTPUT=speakers, GAIN=0)
	uv run jammate --headless --input "$(INPUT)" --output "$(OUTPUT)" \
		--input-channel $(CHANNEL) --block-size $(BLOCK) --monitor-gain $(GAIN) \
		--instrument $(INSTRUMENT) --model-size $(MODEL)

capture: ## Record raw input to FILE (optional DURATION in s), e.g. FILE=tests/data/riff.wav DURATION=20
	$(if $(FILE),,$(error FILE is required, e.g. make capture FILE=tests/data/riff.wav DURATION=20))
	uv run python scripts/capture_input.py "$(FILE)" --input "$(INPUT)" --output "$(OUTPUT)" \
		--input-channel $(CHANNEL) --monitor-gain $(GAIN) $(if $(DURATION),--seconds $(DURATION))

analyze: ## Print the analysis event timeline of FILE (a WAV), e.g. FILE=tests/data/guitar_open_strings.wav
	$(if $(FILE),,$(error FILE is required, e.g. make analyze FILE=tests/data/guitar_open_strings.wav))
	uv run python scripts/analyze_wav.py "$(FILE)" --instrument $(INSTRUMENT)

render: ## Render an offline jam from FILE to OUT (default out.wav), e.g. FILE=tests/data/guitar_riff_e2_g2_a2_120bpm.wav
	$(if $(FILE),,$(error FILE is required, e.g. make render FILE=tests/data/guitar_riff_e2_g2_a2_120bpm.wav OUT=out.wav))
	uv run python scripts/render_jam.py "$(FILE)" "$(OUT)" --instrument $(INSTRUMENT) --style $(STYLE) --model-size $(MODEL)

app: ## Build the standalone macOS JamMate.app bundle (Step 7.2)
	uv run python scripts/build_macos_app.py

mrt-check: ## Generate 4 s with Magenta RT2 (MODEL=mrt2_small or mrt2_base) to check the model install
	uv run mrt mlx generate --prompt "$(PROMPT)" --duration 4.0 --model=$(MODEL)

clean: ## Remove Python, pytest and ruff caches (keeps .venv)
	rm -rf .pytest_cache .ruff_cache
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +
