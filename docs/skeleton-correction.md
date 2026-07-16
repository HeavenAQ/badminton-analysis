# Skeleton-Correction Scoring

The first implementation is intentionally limited to badminton clear. It reuses
the existing pose extraction and analysis-window detection, normalizes the full
window to 64 frames, and trains a denoising Transformer on expert sequences.

## Data contract

Each file under `datasets/skeleton_sequences/clear/{beginners,experts}/` is an
NPZ archive containing:

- `skeleton_3d`: normalized `64 x 17 x 3` COCO skeleton
- `skeleton_2d`: normalized `64 x 17 x 2` COCO skeleton
- `confidence`: observed-joint mask resampled to `64 x 17`
- `skill`, `handedness`, and `video_name`
- `analysis_window`: source `(start, peak, end)` tracked-frame indices
- `phase_indices`: five key-frame indices in the resampled window
- `fps`: source video frame rate when available

Left-handed anatomy is swapped into right-dominant COCO slots before poses are
pelvis-centered, rotated into the preparation frame's fixed body basis, and
scaled by median shoulder width. The fixed basis retains torso rotation through
the clear while removing camera orientation differences.
Coordinates are interpolated across tracking gaps, but `confidence` continues
to identify which joints were actually observed.

Handedness is estimated from the complete tracked clip before selecting the
analysis window. Both wrist trajectories are centered on the torso, scaled by
shoulder width, smoothed, and scored from their robust peak speed and positive
acceleration. The higher-motion wrist is accepted only when its score is at
least twice the other wrist's score. Ambiguous clips fall back to
`datasets/skeleton_sequences/clear/handedness_overrides.json`, then to filename
metadata. This avoids forcing a handedness decision from tracking noise.

## Run the clear feasibility workflow

Extract both labeled groups:

```bash
.venv/bin/python scripts/extract_skeleton_sequences.py
```

Train the expert reconstruction and student-to-expert correction objectives:

```bash
.venv/bin/python -m badminton_analysis.ml.train_skeleton_corrector \
  --epochs 150
```

Score both groups and compare them with baseline CSVs:

```bash
.venv/bin/python -m badminton_analysis.ml.infer_skeleton_corrector \
  --baseline-beginners stats/skeleton_correction/baseline_clear_beginners/grading_results.csv \
  --baseline-experts stats/skeleton_correction/baseline_clear_experts/grading_results.csv
```

Render the four required debug cohorts:

```bash
.venv/bin/python scripts/render_skeleton_correction_overlays.py
```

Training writes the best-validation checkpoint to
`models/skeleton_correction/clear_expert_guided_v3.pt` and per-epoch component
losses to
`stats/skeleton_correction/clear_expert_guided_v3_training/training_metrics.csv`.
Inference writes
compatible grading rows, raw distance components, score summaries, calibration,
and old/new comparisons under `stats/skeleton_correction/clear_feasibility/`.

## Feasibility gate

Do not enable this backend for other skills solely because calibrated means hit
the requested score bands. Before CLI integration, verify all of the following:

1. Raw correction distances separate experts from beginners with limited overlap.
2. Expert corrections are intrinsically small before score calibration.
3. Low-score overlays show plausible joint and phase corrections without limb
   stretching, joint swaps, or temporal instability.
4. At least 90% of source videos have usable sequences, with failures retained in
   `stats/skeleton_correction/clear_dataset_summary.csv`.

The existing rule/prototype grading path remains the default until this gate is
met.

After calibration has been generated, the clear backend is available through the
compatible grading CLI:

```bash
.venv/bin/python -m badminton_analysis.tools.grade_students \
  --skill clear \
  --input-dir scoring_videos/高遠球/初學者高遠球 \
  --output-dir stats/skeleton_correction/clear_cli_test \
  --scorer skeleton-correction \
  --model-path models/skeleton_correction/clear_expert_guided_v3.pt
```

Omitting `--scorer` preserves the existing behavior. Skeleton correction is
rejected for skills other than clear. Its scores are experimental diagnostics,
not validated grades.

## Baseline recorded on 2026-07-16

The current prototype scorer completed all 100 clear videos without a processing
failure. Beginners averaged 72.93 (median 69.82, range 57.65-94.74), while
experts averaged 92.42 (median 93.89, range 77.37-99.94). Fourteen of 50
beginners, or 28%, scored at or above the lowest expert. This is the distribution
the skeleton-correction experiment must improve without merely lowering every
video.

## Historical v1 experiment result (superseded)

The first correction-distance run completed on 2026-07-16. These values are
retained as experiment history and do not describe the current v3 model:

| Metric | Beginners | Experts | Held-out experts |
|---|---:|---:|---:|
| Videos | 50 | 50 | 10 |
| Mean score | 45.00 | 98.18 | 99.03 |
| Median score | 41.02 | 100.00 | 100.00 |
| Mean raw correction distance | 0.1564 | 0.1089 | 0.1061 |

- All 100 videos produced valid `64 x 17` sequences; mean missing-joint ratio was
  0.00376 and the maximum was 0.03406.
- Held-out expert/beginner correction-distance AUC was 0.966. Eight percent of
  beginners overlapped the held-out expert distance range.
- Beginner scores fell 27.93 points from the prototype baseline on average.
- Training used a deterministic 40/10 expert split. Best corrupted-expert
  validation loss was 0.26047 after 100 epochs.
- Inference applies an iterative pelvis-anchored bone-length projection. Across
  the final results, the worst per-video p95 relative bone-length change was less
  than 0.006%, removing the stretching visible in the first unconstrained model.
- No limb stretching or joint swaps were observed in the four review cohorts.
  However, qualitative review did not support the severity of some low grades.
- The initial run assumed right-handed source clips. The later full-clip wrist
  motion pass identified `EG28` and `EG29` as left-handed with motion-score
  ratios of `6.41` and `6.58`; both were re-extracted and included in the
  subsequent retraining.

The denoising Transformer does **not** yet meet the clear feasibility gate. In
particular, `EG34.mp4` received `4.46` from correction distance while the prior
scorer gave it `81.96`, and the overlaid correction was not visually severe enough
to justify the new grade. The exponential mapping was fitted to force the
beginner/expert group means to target bands; it was not fitted to per-video human
grades. This can create extreme scores that are mathematically consistent with
the group calibration but technically unjustified for an individual stroke.

Before this backend is used for grading, collect human scores for individual
clips, fit the distance-to-grade mapping on those scores, and validate held-out
per-video error and rank correlation. Until then, use the raw correction
components and overlays for model diagnostics only. Do not extend the model to
other skills.

## Full expert-reference retraining

The v2 objective only moved a student 50% toward an expert during training and
applied 80% of the learned residual during inference. That design could improve
distance without producing the expert-like corrected skeleton required here, so
v2 is superseded.

The v3 pipeline:

1. Splits both experts and students into independent 35/5/10
   train/validation/test sets.
2. Warps each sequence onto shared preparation, rotation, contact, and
   follow-through phase anchors.
3. Selects the nearest training expert and adapts the complete expert pose to the
   student's pelvis position and limb lengths. There is no partial blend.
4. Conditions the seven-feature Transformer on student XYZ, the adapted expert
   XYZ reference, and confidence, then applies the full predicted correction.
5. Restores only the correction delta to source timing and projects the result
   back onto the student's bone-length constraints.
6. Accepts a checkpoint only when every validation student improves, every
   corrected validation student is inside the validation experts' leave-one-out
   95th-percentile distance, and model-to-reference distance is at most `0.10`.
7. Evaluates once on untouched students against untouched experts, while using
   only training experts as correction references.

The unconditioned full-target run plateaued with only 80% of validation students
inside the expert range and was rejected. The accepted reference-conditioned
checkpoint is epoch 54 of
`models/skeleton_correction/clear_expert_guided_v3.pt`. Audit files are under
`stats/skeleton_correction/clear_expert_guided_v3_training/`:

| Metric | Validation | Untouched test |
|---|---:|---:|
| Students | 5 | 10 |
| Expert-range threshold | 0.2920 | 0.3203 |
| Input nearest-expert distance | 0.4714 | 0.4209 |
| Corrected nearest-expert distance | 0.2307 | 0.1911 |
| Corrected inside expert range | 100% | 100% |
| Mean model-to-reference distance | 0.0438 | 0.0374 |
| Maximum model-to-reference distance | 0.0614 | 0.0488 |

For context, untouched experts have a mean leave-one-out nearest-expert distance
of `0.2456`. Across all 50 students, the worst model-to-reference distance is
`0.0614`, the mean correction acceleration is `0.0373`, pelvis drift is zero,
and the worst p95 relative bone-length change is below `0.00024%`.

`EG12` is the untouched student with the largest input distance. Its distance to
untouched experts changes from `0.5426` to `0.1966`, and its model output is
`0.0423` from the selected bone-adapted training-expert target. The reviewed
overlay video is
`stats/skeleton_correction/clear_debug_videos/EG12_expert_guided_v3_test_h264.mp4`.

These checks validate expert-like skeleton geometry, not a numeric badminton
grade. Per-video human scores are still required before correction magnitude can
be calibrated into a defensible grade.

## Complete scored review set

Generate group-calibrated diagnostic grades for all 50 students and 50 experts:

```bash
.venv/bin/python -m badminton_analysis.ml.infer_skeleton_corrector \
  --model-path models/skeleton_correction/clear_expert_guided_v3.pt \
  --output-dir stats/skeleton_correction/clear_expert_guided_v3_grades
```

Render and H.264-transcode all 100 scored overlay videos while reusing the loaded
pose and correction models:

```bash
.venv/bin/python scripts/render_all_skeleton_correction_videos.py \
  --output-dir stats/skeleton_correction/clear_expert_guided_v3_videos
```

The completed review set contains:

```text
stats/skeleton_correction/clear_expert_guided_v3_videos/students/  # 50 videos
stats/skeleton_correction/clear_expert_guided_v3_videos/experts/   # 50 videos
stats/skeleton_correction/clear_expert_guided_v3_videos/render_summary.csv
stats/skeleton_correction/clear_expert_guided_v3_grades/all_grades.csv
stats/skeleton_correction/clear_expert_guided_v3_grades/grading_results.csv
stats/skeleton_correction/clear_expert_guided_v3_grades/keypoint_scores.csv
stats/skeleton_correction/clear_expert_guided_v3_grades/advice_context.jsonl
```

The current v3 correction-distance export contains 50 students with mean score
`45.00` and 50 experts with mean score `99.90`. Run
`scripts/export_clear_correction_scores.py` to write the full per-video CSV and
the group means from `grading_results.csv`.

Each video header shows the subject name and the matching CSV grade. The compact
`all_grades.csv` marks every row as `diagnostic_group_calibrated`; use the scores
for comparative review, not as human-validated badminton grades.

## Keypoint-grounded coaching output

`keypoint_scores.csv` has one row per subject and coaching keypoint. Dominant
and non-dominant names refer to the player's racket side after handedness
normalization, so `dominant_wrist` is the left wrist for `EG28` and `EG29` even
though it occupies the canonical dominant-side model slot. Each row includes:

- the keypoint's calibrated diagnostic score and attributed correction distance;
- position, angle, velocity, and bone-length distance components;
- preparation, rotation, contact, and follow-through scores and distances;
- the worst phase, correction direction, and mean 3D correction vector; and
- the joint importance weight used by the sequence metric.

`advice_context.jsonl` groups the same evidence into one JSON object per person
and ranks the five highest-priority corrections. It is the preferred input for
a language model generating advice. The model should mention only corrections
supported by those fields, use `worst_phase` to say when they occur, and use
`correction_direction` to describe the requested change.

The total grade remains the calibrated weighted correction over the complete
sequence. Keypoint scores use the same calibration on joint-attributed distance
and explain which joints contributed evidence; they are not independent rubric
points and must not be added to reconstruct the total grade. As with the total
score, these values are diagnostic until calibrated against per-video human
grades and coaching labels.

## OpenAI-assisted coaching video

The current OpenAI vision interface accepts one or more ordered image inputs,
while the analysis models do not accept an MP4 as a video modality. The coaching
pass therefore samples 11 checkpoint-aware, timestamped frames from the
64-frame review clip, including all five stored grading checkpoints, and sends
them together in one Responses API request. This follows the
[OpenAI image-input guidance](https://developers.openai.com/api/docs/guides/images-vision),
which supports multiple Base64 image inputs in a request.

Run a structured clear analysis using `OPENAI_API_KEY` from `.env`:

```bash
.venv/bin/python scripts/analyze_clear_with_openai.py \
  --video-path stats/skeleton_correction/clear_expert_guided_v3_videos/students/EG29.mp4 \
  --output-dir stats/openai_clear_feedback/EG29
```

The request includes handedness, the exact six `ClearGrader` criterion names and
formulas, current expert means and standard deviations, student measurements at
the five grading checkpoints without legacy display scores, the calibrated
correction-distance total and six allocations, keypoint evidence, and the
overlay legend. The response schema requires `zh-TW`, restricts every title to
one of the six original Traditional Chinese criterion names, and permits only
that criterion's grading frame and measured canonical joint IDs. It produces:

```text
stats/openai_clear_feedback/EG29/input_frames/
stats/openai_clear_feedback/EG29/prompt_context.json
stats/openai_clear_feedback/EG29/feedback.json
stats/openai_clear_feedback/EG29/feedback.csv
```

Render the feedback onto the corrected video:

```bash
.venv/bin/python scripts/render_skeleton_correction_video.py \
  --video-path scoring_videos/高遠球/初學者高遠球/EG29.mp4 \
  --dataset-path datasets/skeleton_sequences/clear/beginners/EG29.npz \
  --feedback-path stats/openai_clear_feedback/EG29/feedback.json \
  --output-path stats/openai_clear_feedback/EG29/annotated_feedback.mp4
```

Each reported problem freezes its selected frame for two seconds, circles the
requested detected joints, and displays the short Traditional Chinese coaching
instruction and criterion score. The nameplate and six criterion allocations
come from the calibrated difference between the original and corrected
skeletons. The original `ClearGrader` criteria and checkpoints constrain the
coaching language, but their legacy angle-formula totals are not displayed.
The corrected live `EG29` pass selected the original
`慣用手肩膀往前轉` criterion at final follow-through frame 63 and canonical
dominant shoulder joint 6. The H.264 review artifact is
`stats/openai_clear_feedback/EG29/annotated_feedback_h264.mp4`.

The complete implementation trace and other-skill extension checklist are in
`docs/skeleton-correction-pipeline.md`.
