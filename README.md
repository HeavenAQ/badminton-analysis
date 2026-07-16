# Badminton Analysis Module

Pose extraction, angle computation, and rule-based / GPT-vision skill grading for badminton analysis.

## Current Progress

**Implemented and working:**
- Pose extraction from video via YOLO-based `VideoProcessor`
- Analysis window detection via `VideoAnalyzer` (acceleration-based for serve/lift; lowest-hand-position-based for smash/clear)
- Angle computation for 10 joint features (shoulder, elbow, knee, crotch, nose–shoulder–elbow)
- Left/right → dominant/non-dominant normalization via label mirroring (`mirror_angles`) before stats aggregation
- Rule-based graders for: **serve**, **smash**, **lift**, **forehand/backhand drive**, **forehand/backhand net kill**, **footwork** (DTW)
- GPT-vision grader (`scripts/grade_students_gpt.py`) as an alternative to the rule-based pipeline — sends expert stats, reference images, and key frames to GPT-4.1
- Statistical analysis scripts (`scripts/descrptive_analysis.py`) producing mean, std, CV, mean-|z| CSVs and two visualizations: temporal angle profiles and a CV heatmap
- Student grading CLI: `grade-students` (outputs per-video CSV with total score and per-checkpoint breakdown)

## Skill Grading Checkpoints

Each skill's analysis window is divided into 5 key frames:

| Index | Position |
|---|---|
| 0 | Start of window |
| 1 | Midpoint(start, peak) |
| 2 | Peak |
| 3 | Midpoint(peak, end) |
| 4 | End of window |

### Lift (3 checkpoints, 100 pts)

| # | Name | English | Frame(s) | Pts | Joints |
|---|---|---|---|---|---|
| 1 | 手腕放置腰部放鬆預備 | Wrist at waist, relaxed ready | 0 | 10 | Dom. Shoulder (5), Non-dom. Shoulder (5) |
| 2 | 手腕往後引拍 | Draw wrist back — backswing | 2 | 45 | Dom. Shoulder (22.5), Dom. Elbow (22.5) |
| 3 | 手腕往前壓 | Press wrist forward — snap | 3 | 45 | Dom. Shoulder (22.5), Dom. Elbow (22.5) |

### Smash (6 checkpoints, 100 pts)

| # | Name | English | Frame(s) | Pts | Joints / criterion |
|---|---|---|---|---|---|
| 1 | 球拍舉至腰部預備 | Racket raised to waist | 0 | 10 | Dom. Shoulder (5), Non-dom. Shoulder (5) |
| 2 | 轉身 | Body rotation | 0 → 1 | 10 | Non-dom. ankle X − dom. ankle X change > 10 px (binary) |
| 3 | 雙手手肘平衡 | Both elbows balanced | 1 | 20 | Dom. Shoulder (10), Non-dom. Shoulder (10) |
| 4 | 手肘往前轉至前方 | Dominant elbow drives forward | 2 | 20 | Dom. Shoulder (20) |
| 5 | 手腕發力 | Wrist flick | 2 | 20 | Dom. Elbow (20) |
| 6 | 慣用手肩膀往前轉 | Dominant shoulder rotates forward | 3 | 20 | Dom. Shoulder angle (10) + dom. shoulder X − non-dom. shoulder X > 5 px (10, binary) |

### Serve (5 checkpoints / 6 sub-checks, 100 pts)

| # | Name | English | Frame(s) | Pts | Joints / criterion |
|---|---|---|---|---|---|
| 1a | 雙手平舉 | Both arms raised | 0 | 10 | Dom. Shoulder (5, if ≥ 25°), Non-dom. Shoulder (5, if ≥ 25°) |
| 1b | 將重心放至持拍腳 | Weight on racket foot | 0 | 10 | Dom. Crotch ≤ Non-dom. Crotch (binary) |
| 2 | 重心轉移至非持拍腳 | Weight transfer to non-racket foot | 0 → 1 (lower), 1 → 3 (upper) | 20 | min(crotch displacement, eye/ear X displacement) |
| 3 | 髖關節前旋 | Hip-axis rotation | 0 → 4 | 20 | Hip-axis angle change vs expert mean |
| 4 | 持拍手手腕發力 | Dominant wrist flick | 3 | 20 | Dom. Elbow (20) |
| 5 | 肩膀旋轉朝前 | Shoulder rotates forward | 4 | 20 | Dom. Shoulder (10), Dom. Shoulder–Elbow (10) |

### Drive — forehand and backhand (3 checkpoints, 100 pts)

| # | Name | English | Frame(s) | Pts | Joints |
|---|---|---|---|---|---|
| 1 | 手腕放置腰部放鬆預備 | Wrist at waist, relaxed ready | 0 | 10 | Dom. Shoulder (5), Non-dom. Shoulder (5) |
| 2 | 手腕往後引拍 | Draw wrist back — backswing | 2 | 45 | Dom. Shoulder (22.5), Dom. Elbow (22.5) |
| 3 | 手腕往前壓 | Press wrist forward | 3 | 45 | Dom. Shoulder (22.5), Dom. Elbow (22.5) |

### Net Kill — forehand and backhand (3 checkpoints, 100 pts)

| # | Name | English | Frame(s) | Pts | Joints |
|---|---|---|---|---|---|
| 1 | 手腕放置腰部放鬆預備 | Wrist at waist, relaxed ready | 0 | 10 | Dom. Shoulder (5), Non-dom. Shoulder (5) |
| 2 | 手腕往後引拍 | Draw wrist back — backswing | 2 | 45 | Dom. Shoulder (22.5), Dom. Elbow (22.5) |
| 3 | 手腕往前壓 | Press wrist forward | 3 | 45 | Dom. Shoulder (22.5), Dom. Elbow (22.5) |

## Usage

### Skeleton correction (clear feasibility)

The experimental clear-only skeleton-correction workflow is documented in
[`docs/skeleton-correction.md`](docs/skeleton-correction.md). It extracts fixed
64-frame dominant-side skeleton sequences, trains an expert denoiser, evaluates
correction-distance separation, and renders correction overlays. The quantitative
experiment separates the labeled groups, but the per-video grades have not passed
qualitative validation and must be treated as diagnostic only. The backend is
opt-in with `--scorer skeleton-correction`; the existing default remains unchanged.
The expert-guided v3 checkpoint uses a full, bone-preserving training-expert
target and rejects checkpoints unless every validation correction is inside the
natural expert-to-expert Euclidean distance range. All ten untouched test
students passed the same test. This validates correction geometry, but it does
not replace the missing human grade calibration. Full-clip wrist motion now
determines handedness before phase extraction, with conservative override
fallbacks for ambiguous clips. Inference also writes per-keypoint and per-phase
correction evidence plus ranked JSONL context for language-model coaching; those
signals remain diagnostic rather than human-validated rubric scores.

`scripts/analyze_clear_with_openai.py` turns one scored clear clip into an
ordered image sequence, combines it with the clear rules and keypoint evidence,
and requests Traditional Chinese timestamp/joint feedback from the OpenAI
Responses API. Titles, frames, and joint IDs are constrained to the exact six
`ClearGrader` criteria and their original grading checkpoints.
The video renderer can then pause at each selected frame, circle those joints,
and display the coaching instruction. See the OpenAI-assisted coaching section
in [`docs/skeleton-correction.md`](docs/skeleton-correction.md). For a complete
code-level call graph and the required replacement points for another skill, see
[`docs/skeleton-correction-pipeline.md`](docs/skeleton-correction-pipeline.md).

### With uv (recommended)

- Sync dependencies (project + dev): `uv sync --dev`
- Run tests: `uv run -m pytest -q`
- Type check: `uv run -m mypy . --config-file mypy.ini`
- Batch analyze: `uv run -m badminton_analysis.tools.analyze --input training_videos --output stats`
- Grade students: `uv run -m badminton_analysis.tools.grade_students --handedness right --skill serve --input-dir student_videos --output-dir grading_results`
- Grade footwork: `uv run -m badminton_analysis.tools.grade_students --handedness right --skill footwork --input-dir student_videos --output-dir grading_results --reference-data footwork_reference.json`

Notes:
- Default dependency is `opencv-python-headless`. For GUI windows, install extras: `uv pip install .[gui]` or `uv add '.[gui]'`.
- Dev tools are also available as extras: `uv add '.[dev]'`.
- If you see a uv warning about an active virtual environment, simply deactivate any venv you’ve previously activated (`deactivate`) and rerun. uv manages the project’s `.venv` automatically, and the Makefile clears `VIRTUAL_ENV`/`CONDA_PREFIX` for uv commands to avoid this warning.

Footwork reference JSON format:

```json
{
  "right": {
    "left_ankle": [[0.0, 0.0], [0.2, 0.1]],
    "right_ankle": [[0.0, 0.0], [0.3, 0.0]]
  },
  "left": {
    "left_ankle": [[0.0, 0.0], [0.1, 0.2]],
    "right_ankle": [[0.0, 0.0], [0.0, 0.3]]
  }
}
```

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
  - Golden-frame fixtures for step events and acceptable angle ranges.
  - CLI: batch extraction plus student grading flows.

- Visualization
  - Overlay detected direction labels and foot placement markers.
  - Export per‑frame annotations alongside the output video.

## Dependencies

See `requirements.txt`. OpenCV/MediaPipe/Torch/Ultralytics are required for pose and visualization.
