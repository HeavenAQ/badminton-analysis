# Badminton Skeleton Correction

This repository contains the current clear-only badminton analysis pipeline. It
extracts 2D and 3D skeletons from video, predicts an expert-like corrected
skeleton, scores the required correction, and produces Traditional Chinese
coaching videos with timestamped joint annotations.

The old rule graders, prototype scorers, statistical scripts, and annotation
utilities are not part of this version.

## What The Pipeline Produces

For each clear video, the pipeline can produce:

- a normalized 64-frame 3D skeleton sequence;
- an expert-guided corrected skeleton;
- original-to-corrected position, angle, velocity, and bone-length distances;
- a calibrated total score and six criterion allocations;
- per-joint and per-phase evidence for language-model coaching;
- an overlay video with detected and corrected skeletons; and
- Traditional Chinese feedback with automatic pauses and joint circles.

The committed model is:

```text
models/skeleton_correction/clear_expert_guided_v3.pt
models/skeleton_correction/clear_expert_guided_v3.calibration.json
```

## Architecture

```text
video
  -> RTMW whole-body 2D pose
  -> MMPose 3D lifting
  -> handedness from normalized wrist acceleration
  -> clear analysis window and five phase anchors
  -> dominant-side normalization and 64-frame resampling
  -> nearest phase-aligned training expert by masked 3D Euclidean distance
  -> expert reference adapted to the student's bone lengths
  -> reference-conditioned Transformer correction
  -> geometry and expert-distance validation
  -> correction-distance score and keypoint evidence
  -> optional OpenAI coaching and annotated video
```

The full code-level trace is in
[`docs/skeleton-correction-pipeline.md`](docs/skeleton-correction-pipeline.md).

## Scoring

The score measures how much the original skeleton must move to reach the
model-predicted expert-like skeleton:

```text
D = position + 0.5 * angle + 0.5 * velocity + 0.25 * bone_length

score(D) = 100 * exp(-alpha * max(D - offset, 0))
```

The committed clear calibration uses:

```text
offset = 0.24125837235747438
alpha  = 2.680872933195534
```

On the current 50-student and 50-expert dataset, the calibrated means are 45.00
for students and 99.90 for experts. These are group-fitted diagnostic scores,
not independently validated human grades. Production grading requires
per-video human labels and held-out calibration validation.

## Requirements

- Python 3.12
- PyTorch
- OpenCV
- pandas, NumPy, Pillow, Pydantic, and the OpenAI Python SDK
- a compatible OpenMMLab pose stack: `mmengine`, `mmcv`, `mmdet`, and `mmpose`
- FFmpeg for H.264 review videos

Install the Python project dependencies:

```bash
uv sync --dev
```

Install the OpenMMLab packages separately for the target PyTorch/CUDA
environment. They are imported lazily, so unit tests and score-processing tools
can run without loading pose models.

For OpenAI coaching, place `OPENAI_API_KEY` in `.env`.

## Data Layout

Source videos and generated artifacts are intentionally ignored by Git.
Extraction creates:

```text
datasets/skeleton_sequences/clear/
  beginners/*.npz
  experts/*.npz
  handedness_overrides.json
```

Each NPZ contains `skeleton_3d`, `skeleton_2d`, `confidence`, handedness,
analysis-window indices, five phase indices, video name, and FPS.

## Extract Skeleton Sequences

```bash
.venv/bin/python scripts/extract_skeleton_sequences.py \
  --beginner-dir /path/to/student/clear/videos \
  --expert-dir /path/to/expert/clear/videos \
  --output-root datasets/skeleton_sequences/clear
```

Handedness is inferred from full-clip left/right wrist motion. Ambiguous clips
can be specified in `handedness_overrides.json`.

## Train

```bash
.venv/bin/python -m badminton_analysis.ml.train_skeleton_corrector \
  --dataset-root datasets/skeleton_sequences/clear \
  --model-path models/skeleton_correction/clear_expert_guided_v3.pt \
  --epochs 150
```

Training uses independent expert and student train/validation/test splits. A
checkpoint is accepted only when validation corrections improve toward experts,
enter the expert distance range, preserve bone lengths, remain temporally
stable, and stay close to their selected expert references.

## Evaluate And Generate Scores

```bash
.venv/bin/python -m badminton_analysis.ml.infer_skeleton_corrector \
  --dataset-root datasets/skeleton_sequences/clear \
  --model-path models/skeleton_correction/clear_expert_guided_v3.pt \
  --output-dir stats/skeleton_correction/clear_expert_guided_v3_grades

.venv/bin/python scripts/export_clear_correction_scores.py
```

Important outputs include:

```text
grading_results.csv
distance_components.csv
score_summary.csv
calibration.json
keypoint_scores.csv
advice_context.jsonl
```

## Score A Directory Of Clear Videos

```bash
.venv/bin/python -m badminton_analysis.tools.grade_students \
  --input-dir /path/to/clear/videos \
  --output-dir stats/clear_scores \
  --handedness auto
```

Use `--handedness right` or `--handedness left` when automatic wrist-motion
evidence is ambiguous.

## Render Skeleton Corrections

Render all students and experts:

```bash
.venv/bin/python scripts/render_all_skeleton_correction_videos.py \
  --dataset-root datasets/skeleton_sequences/clear \
  --model-path models/skeleton_correction/clear_expert_guided_v3.pt \
  --output-dir stats/skeleton_correction/clear_expert_guided_v3_videos
```

Render one source video:

```bash
.venv/bin/python scripts/render_skeleton_correction_video.py \
  --video-path /path/to/EG3.mp4 \
  --dataset-path datasets/skeleton_sequences/clear/beginners/EG3.npz \
  --output-path stats/EG3_corrected.mp4
```

The detected skeleton is cyan and the corrected skeleton is green.

## Generate LLM Coaching

First generate structured Traditional Chinese feedback:

```bash
.venv/bin/python scripts/analyze_clear_with_openai.py \
  --video-path stats/skeleton_correction/clear_expert_guided_v3_videos/students/EG3.mp4 \
  --dataset-path datasets/skeleton_sequences/clear/beginners/EG3.npz \
  --advice-path stats/skeleton_correction/clear_expert_guided_v3_grades/advice_context.jsonl \
  --grading-results-path stats/skeleton_correction/clear_expert_guided_v3_grades/grading_results.csv \
  --output-dir stats/openai_clear_feedback/EG3
```

Then render the annotations:

```bash
.venv/bin/python scripts/render_skeleton_correction_video.py \
  --video-path /path/to/EG3.mp4 \
  --dataset-path datasets/skeleton_sequences/clear/beginners/EG3.npz \
  --feedback-path stats/openai_clear_feedback/EG3/feedback.json \
  --output-path stats/openai_clear_feedback/EG3/annotated_feedback.mp4
```

The renderer pauses at each reported checkpoint, circles deterministic coaching
joints, maps dominant joints back to the physical left or right side, and shows
the correction-distance score beside the student name.

## Tests

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m mypy \
  badminton_analysis scripts tests \
  --config-file mypy.ini
```

The focused tests cover handedness, normalization, phase alignment, nearest
expert selection, bone projection, score monotonicity, calibration, coaching
schema validation, physical-side joint mapping, pose conversion, and CLI output.

## Current Scope

Only overhead clear is supported by the correction model and CLI. Adding another
skill requires a separate phase contract, expert reference bank, joint/phase
weights, checkpoint, calibration, and coaching criteria. Do not reuse the clear
calibration for another skill.
