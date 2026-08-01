# Badminton Skeleton Correction

This repository implements expert-guided skeleton correction for four badminton
skills:

- `serve` - serve;
- `lift` - lift;
- `clear` - overhead clear; and
- `smash` - overhead smash.

Every skill has a separate dataset, expert reference bank, checkpoint,
calibration, joint weighting, qualitative criteria, and output directory. The
skills share pose extraction, normalization, Transformer correction, score
calibration, and rendering code, but training examples and scores are never
pooled across skills.

Accepted checkpoints, ONNX exports, and calibrations are committed for all four
skills:

```text
models/skeleton_correction/clear_expert_guided_v3.pt
models/skeleton_correction/clear_expert_guided_v3.calibration.json
models/skeleton_correction/serve_expert_guided_v1.pt
models/skeleton_correction/lift_expert_guided_v1.pt
models/skeleton_correction/smash_expert_guided_v1.pt
```

Each skill is trained and calibrated independently.

## Expert Data Contract

The legacy expert corpus contains 50 right-handed experts for clear, lift, and
smash. It has no left-handed experts. Serve intentionally excludes the legacy
`scoring_videos/發球/羽球隊同學` directory. NSTC contributes experts only from
these exact directories:

```text
training_videos/nstc/{clear,serve,lift,smash}/left
training_videos/nstc/{clear,serve,lift,smash}/right
```

Person-named directories below a skill are intentionally excluded. NSTC
sequences use `nstc_left_` or `nstc_right_` ID prefixes to prevent same-named
videos in the two hand directories from colliding. The combined expert counts
are:

| Skill | Right | Left | Total |
|---|---:|---:|---:|
| Clear | 80 | 20 | 100 |
| Serve | 16 | 10 | 26 |
| Lift | 72 | 10 | 82 |
| Smash | 66 | 9 | 75 |

## Outputs

For one selected skill, the pipeline produces:

- a dominant-side-normalized 64-frame 2D/3D skeleton sequence;
- original-video frame provenance for every normalized frame;
- the nearest phase-aligned training expert and bone-length-adapted reference;
- an expert-like corrected skeleton;
- position, angle, velocity, and bone-length correction distances;
- for serve, separate full-sequence lower-support and torso-forward-lean distances;
- a cohort-calibrated diagnostic score and criterion allocations;
- per-joint and per-phase evidence;
- detected/corrected skeleton videos; and
- optional Traditional Chinese LLM feedback with pauses and joint circles.

## Skill Criteria

The qualitative contracts come from the previous skill implementations. They
constrain score allocation and LLM advice; the numeric score itself comes from
the current correction-distance algorithm.

### Serve

| Criterion | Maximum |
|---|---:|
| 雙手平舉 | 10 |
| 將重心放至持拍腳 | 10 |
| 重心轉移至非持拍腳 | 20 |
| 髖關節前旋 | 20 |
| 持拍手手腕發力 | 20 |
| 肩膀旋轉朝前 | 20 |

### Lift

| Criterion | Maximum |
|---|---:|
| 手腕放置腰部放鬆預備 | 10 |
| 手腕往後引拍 | 25 |
| 手腕往前壓 | 35 |
| 手腕放鬆回到預備姿勢 | 30 |

### Clear And Smash

Both retain the six original qualitative checkpoints, but use independent
expert data, checkpoints, calibrations, joint weights, and prompt descriptions.

| Criterion | Maximum |
|---|---:|
| 球拍舉至腰部預備 | 10 |
| 轉身 | 10 |
| 雙手手肘平衡 | 20 |
| 手肘往前轉至前方 | 20 |
| 手腕發力 | 20 |
| 慣用手肩膀往前轉 | 20 |

## Architecture

```text
one skill's source videos
  -> direct RTMW3D whole-body 2D/3D pose
  -> authoritative expert handedness, or wrist acceleration for unlabeled video
  -> skill-specific analysis window
  -> dominant-side/body-frame normalization
  -> 64-frame resampling plus source-frame provenance
  -> five-anchor phase alignment
  -> filter the checkpoint expert bank to the student's handedness
  -> adapt every training expert to the student's bone lengths
  -> nearest expert by skill-weighted grading distance
  -> reference-conditioned temporal/spatial Transformer
  -> learned output blended 50% toward the adapted expert reference
  -> bone-length projection and checkpoint quality gates
  -> skill-weighted correction distance
  -> skill-specific score calibration and criterion evidence
  -> optional OpenAI coaching and annotated video
```

Expert selection has no cross-handed fallback. If the checkpoint has no expert
for the student's handedness, scoring stops with a data-availability error
instead of comparing the student with an opposite-handed motion.

The detailed prerequisites, equations, call graph, and tracing order are in
[docs/skeleton-correction-pipeline.md](docs/skeleton-correction-pipeline.md).

## Requirements

- Python 3.12
- PyTorch, NumPy, pandas, OpenCV, Pillow, Pydantic, and the OpenAI SDK
- RTMLib `0.0.15` and ONNX Runtime `1.24.4`
- NVIDIA TensorRT and `onnxruntime-gpu` for the production GPU provider
- FFmpeg for H.264 review videos

Install project dependencies:

```bash
uv sync --dev
```

The pose detector loads the RTMW3D ONNX weights configured by
`RTMW3D_DETECTOR_MODEL` and `RTMW3D_POSE_MODEL`. Production uses the TensorRT
execution provider; CPU-only development uses the regular ONNX Runtime package.

## Run One Skill Separately

The following example uses lift. Replace `lift` and the source directories to
run serve or smash.

### 1. Extract

```bash
.venv/bin/python scripts/extract_skeleton_sequences.py \
  --skill lift \
  --beginner-dir /path/to/lift/students \
  --expert-dir /path/to/lift/experts
```

The default output is `datasets/skeleton_sequences/lift/`.

For known expert data, always supply its authoritative handedness. For example:

```bash
.venv/bin/python scripts/extract_skeleton_sequences.py \
  --skill lift \
  --groups experts \
  --expert-dir training_videos/nstc/lift/left \
  --known-handedness left \
  --id-prefix nstc_left_
```

### 2. Train

```bash
.venv/bin/python -m badminton_analysis.ml.train_skeleton_corrector \
  --skill lift
```

The model is accepted only when the student corrections improve toward held-out
experts, enter the expert range, preserve bone lengths, and satisfy correction
magnitude and temporal-smoothness gates.

### 3. Infer And Calibrate

```bash
.venv/bin/python -m badminton_analysis.ml.infer_skeleton_corrector \
  --skill lift
```

This writes grades, calibration, distance components, keypoint evidence, and
LLM advice context under
`stats/skeleton_correction/lift_expert_guided_v1_grades/`.

The displayed score is diagnostic, not an independently validated human grade.
Its per-skill calibration is fitted on the complete evaluated expert and
beginner cohorts, targeting means of `99.8` and `45.0`. Corrected-skeleton
validity is evaluated separately: held-out students must move closer to the
union of permitted training experts and unseen experts, and the report also
retains distance to unseen experts alone.

### 4. Export Scores

```bash
.venv/bin/python scripts/export_correction_scores.py --skill lift
```

### 5. Render All Videos

```bash
.venv/bin/python scripts/render_all_skeleton_correction_videos.py \
  --skill lift \
  --student-video-dir /path/to/lift/students \
  --expert-video-dir /path/to/lift/experts
```

### 6. Generate Traditional Chinese Feedback

For the normalized 64-frame correction video:

```bash
.venv/bin/python scripts/analyze_skill_with_openai.py \
  --skill lift \
  --video-path /path/to/rendered/student.mp4 \
  --video-frame-space normalized \
  --dataset-path datasets/skeleton_sequences/lift/beginners/student.npz
```

Use `--video-frame-space source` when `--video-path` points to the original
video. The extractor's stored frame mapping then selects the correct original
frames.

Render the returned pauses and joint circles:

```bash
.venv/bin/python scripts/render_skeleton_correction_video.py \
  --video-path /path/to/original/student.mp4 \
  --dataset-path datasets/skeleton_sequences/lift/beginners/student.npz \
  --feedback-path stats/openai_lift_feedback/student/feedback.json
```

## Testing

```bash
.venv/bin/pytest -q
```

The tests cover all four skill contracts, handedness, normalization, phase
alignment, expert pairing, correction geometry, score calibration, feedback
schema validation, source-frame provenance, backend output, and rendering
helpers.
