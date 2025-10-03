PY ?= python
PYTEST ?= pytest
MYPY ?= mypy

# Video demo parameters (can be overridden: `make video VIDEO=your.mp4 OUTPUT=out.mp4`)
VIDEO ?= test.mp4
OUTPUT ?= analyzed.mp4

.PHONY: help install type test test-v test-all video clean ci

help:
	@echo "Targets:"
	@echo "  install   - Install Python dependencies (pip + requirements.txt)"
	@echo "  type      - Run mypy static type checks"
	@echo "  test      - Run test suite (quiet)"
	@echo "  test-v    - Run test suite (verbose)"
	@echo "  test-all  - Run type checks, then tests"
	@echo "  video     - Generate analyzed video from $(VIDEO) -> $(OUTPUT)"
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

clean:
	rm -rf __pycache__ .pytest_cache
	rm -f segment.mp4 $(OUTPUT)
	@echo "Cleaned build artifacts and videos"

