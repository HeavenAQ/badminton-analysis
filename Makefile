PY ?= python
PYTEST ?= pytest
MYPY ?= mypy

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
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

type:
	$(MYPY) . --config-file mypy.ini --pretty

test:
	$(PYTEST) -q

test-v:
	$(PYTEST) -vvv

test-all: type test

ci: test-all

video:
	$(PY) main.py $(VIDEO) --output $(OUTPUT)
	@echo "Saved analyzed video to: $(OUTPUT)"

batch:
	$(PY) -m tools.analyze --input "$(BATCH_INPUT)" --output "$(BATCH_OUTPUT)"

clean:
	rm -rf __pycache__ .pytest_cache stats
	rm -f segment.mp4 $(OUTPUT) analyzed.mp4
	@echo "Cleaned build artifacts and videos"
