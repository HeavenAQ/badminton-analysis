PY ?= python
PYTEST ?= pytest
MYPY ?= mypy

# Ensure uv doesn't inherit an already-activated venv (which triggers warnings)
# We clear VIRTUAL_ENV/CONDA_PREFIX for any uv subprocesses.
UV ?= VIRTUAL_ENV= CONDA_PREFIX= uv

BATCH_INPUT ?= training_videos
BATCH_OUTPUT ?= stats
GRADE_INPUT ?= training_videos
GRADE_OUTPUT ?= grading_results
GRADE_HANDEDNESS ?= right
GRADE_SKILL ?= serve
FOOTWORK_REFERENCE ?=

.PHONY: help install type test test-v test-all batch grade clean ci

help:
	@echo "Targets:"
	@echo "  install   - Install Python dependencies (pip + requirements.txt)"
	@echo "  type      - Run mypy static type checks"
	@echo "  test      - Run test suite (quiet)"
	@echo "  test-v    - Run test suite (verbose)"
	@echo "  test-all  - Run type checks, then tests"
	@echo "  batch     - Batch analyze a directory of videos (BATCH_INPUT -> BATCH_OUTPUT)"
	@echo "  grade     - Grade student videos (GRADE_INPUT -> GRADE_OUTPUT)"
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


batch:
	@if command -v uv >/dev/null 2>&1; then \
		$(UV) run -m badminton_analysis.tools.analyze --input "$(BATCH_INPUT)" --output "$(BATCH_OUTPUT)"; \
	else \
		$(PY) -m badminton_analysis.tools.analyze --input "$(BATCH_INPUT)" --output "$(BATCH_OUTPUT)"; \
	fi

grade:
	@if command -v uv >/dev/null 2>&1; then \
		$(UV) run -m badminton_analysis.tools.grade_students --input-dir "$(GRADE_INPUT)" --output-dir "$(GRADE_OUTPUT)" --handedness "$(GRADE_HANDEDNESS)" --skill "$(GRADE_SKILL)" $(if $(FOOTWORK_REFERENCE),--reference-data "$(FOOTWORK_REFERENCE)"); \
	else \
		$(PY) -m badminton_analysis.tools.grade_students --input-dir "$(GRADE_INPUT)" --output-dir "$(GRADE_OUTPUT)" --handedness "$(GRADE_HANDEDNESS)" --skill "$(GRADE_SKILL)" $(if $(FOOTWORK_REFERENCE),--reference-data "$(FOOTWORK_REFERENCE)"); \
	fi

clean:
	rm -rf __pycache__ .pytest_cache stats
	rm -rf grading_results
	rm -f segment.mp4
	@echo "Cleaned build artifacts and videos"
