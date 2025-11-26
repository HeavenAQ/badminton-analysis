PY ?= python
PYTEST ?= pytest
MYPY ?= mypy

# Ensure uv doesn't inherit an already-activated venv (which triggers warnings)
# We clear VIRTUAL_ENV/CONDA_PREFIX for any uv subprocesses.
UV ?= VIRTUAL_ENV= CONDA_PREFIX= uv

# Video demo parameters (can be overridden: `make video VIDEO=your.mp4 OUTPUT=out.mp4`)
VIDEO ?= test.mp4
OUTPUT ?= analyzed.mp4
BATCH_INPUT ?= training_videos
BATCH_OUTPUT ?= stats

.PHONY: help install type test test-v test-all video batch clean ci

help:
	@echo "Targets:"
	@echo "  install   - Install Python dependencies (pip + requirements.txt)"
	@echo "  type      - Run mypy static type checks"
	@echo "  test      - Run test suite (quiet)"
	@echo "  test-v    - Run test suite (verbose)"
	@echo "  test-all  - Run type checks, then tests"
	@echo "  video     - Generate analyzed video from $(VIDEO) -> $(OUTPUT)"
	@echo "  batch     - Batch analyze a directory of videos (BATCH_INPUT -> BATCH_OUTPUT)"
	@echo "  ci        - Same as test-all (convenience for local CI run)"
	@echo "  clean     - Remove caches and generated videos"

install:
	@if command -v uv >/dev/null 2>&1; then \
		echo "Using uv to sync project (prod + dev)"; \
		$(UV) sync --all-extras --dev; \
	else \
		echo "uv not found; falling back to pip + requirements.txt"; \
		$(PY) -m pip install --upgrade pip; \
		$(PY) -m pip install -r requirements.txt; \
	fi

type:
	@if command -v uv >/dev/null 2>&1; then \
		$(UV) run -m mypy . --config-file mypy.ini --pretty; \
	else \
		$(MYPY) . --config-file mypy.ini --pretty; \
	fi

test:
	@if command -v uv >/dev/null 2>&1; then \
		$(UV) run -m pytest -q; \
	else \
		$(PYTEST) -q; \
	fi

test-v:
	@if command -v uv >/dev/null 2>&1; then \
		$(UV) run -m pytest -vvv; \
	else \
		$(PYTEST) -vvv; \
	fi

test-all: type test

ci: test-all


video:
	@if command -v uv >/dev/null 2>&1; then \
		$(UV) run main.py $(VIDEO) --output $(OUTPUT); \
	else \
		$(PY) main.py $(VIDEO) --output $(OUTPUT); \
	fi
	@echo "Saved analyzed video to: $(OUTPUT)"

batch:
	@if command -v uv >/dev/null 2>&1; then \
		$(UV) run -m tools.analyze --input "$(BATCH_INPUT)" --output "$(BATCH_OUTPUT)"; \
	else \
		$(PY) -m tools.analyze --input "$(BATCH_INPUT)" --output "$(BATCH_OUTPUT)"; \
	fi

clean:
	rm -rf __pycache__ .pytest_cache stats
	rm -f segment.mp4 $(OUTPUT) analyzed.mp4
	@echo "Cleaned build artifacts and videos"
