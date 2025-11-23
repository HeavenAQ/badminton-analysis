# Badminton Analysis Module

Pose detection, body‑centric normalization, and joint‑angle visualization for badminton analysis. Includes a test suite and foundations for skill/footwork graders.

## Current Progress

Highlights distilled from recent git history and code state:

- Project initialized and legacy angle‑analysis code integrated with tests (2025‑08‑10).
- Added CI workflows for PR assistant and code review (2025‑08‑10).
- Major refactors to reduce redundancy and speed up calculations (2025‑08‑21).
- Unit tests added and expanded across modules (2025‑08‑22).
- Implemented body‑centric coordinate normalization and initial README (2025‑08‑23).
- Grader framework refactored for future graders (2025‑08‑23).
- Logging added and critical bugs fixed (2025‑08‑23).
- Separated landmark storage for visualization vs. analysis (2025‑09‑06).
- Improved type annotations and general code quality (2025‑09‑06).
- Reorganized files in preparation for future updates (2025‑09‑06).

Functional status right now:

- Live skeleton + angle visualization via `main.py` (OpenCV UI, optional output video).
- Body‑centric normalization available via `Normalizer.py` and covered by tests.
- Core modules in place: `PoseModule.py`, `VideoProcessor.py`, `Grader*.py`, `Logger.py`.
- Initial grader scaffolding for skills (e.g., serve) and footwork.

Sample for running locally: `test.mp4` at the repo root.

## Usage

### With uv (recommended)

- Sync dependencies (project + dev): `uv sync --dev`
- Run tests: `uv run -m pytest -q`
- Type check: `uv run -m mypy . --config-file mypy.ini`
- Live analysis: `uv run main.py test.mp4 --output analyzed.mp4`
- Batch analyze: `uv run -m tools.analyze --input training_videos --output stats`

Notes:
- Default dependency is `opencv-python-headless`. For GUI windows, install extras: `uv pip install .[gui]` or `uv add '.[gui]'`.
- Dev tools are also available as extras: `uv add '.[dev]'`.

Run live analysis with overlayed skeleton and angle arcs:

```
python main.py test.mp4 --output analyzed.mp4
```

Controls: SPACE to pause/resume, Q/ESC to quit.

## TODO (Data‑Driven)

Ground the work in the actual direction subfolders found under `training_videos/*/`:

- Directions Checklist
  - [ ] 右前
  - [ ] 右中
  - [ ] 右後
  - [ ] 左前
  - [ ] 左中
  - [ ] 左後

- Per‑direction tasks
  - Define region features using normalized landmarks (centroid, reach, stance width, orientation).
  - Implement detection/labeling for each direction and validate on clips.
  - Model transitions: origin→direction and direction→origin; capture steps and timing.
  - Handle handedness mapping (left vs. right‑handed) consistently across directions.

- Graders & evaluation
  - Extend `FootworkGrader` with checks specific to each direction (balance, path, recovery).
  - Expand `ServeGrader` separately; keep it orthogonal to direction tasks.
  - Add batch evaluation over `training_videos/` with CSV/Excel summaries.

- Tooling & tests
  - Unit tests per direction and handedness with short fixtures.
  - Golden‑frame fixtures for step events and acceptable angle ranges.
  - CLI: `--batch training_videos` plus per‑direction filters.

- Visualization
  - Overlay detected direction labels and foot placement markers.
  - Export per‑frame annotations alongside the output video.

## Dependencies

See `requirements.txt`. OpenCV/MediaPipe/Torch/Ultralytics are required for pose and visualization.
