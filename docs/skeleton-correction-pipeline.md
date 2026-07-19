# Skeleton-Correction Pipeline: Code Trace and Skill Extension Guide

This document describes the current clear pipeline from source video to
corrected skeleton, calibrated score, criterion-level evidence, and Traditional
Chinese coaching video. It is intended as a code-tracing guide and as the
reference for implementing the same design for another badminton skill.

The central distinction is:

- the **technical criteria** are encoded in `clear_feedback.py` and determine
  what the coach talks about;
- the **numeric score** comes from the difference between the original skeleton
  and the model's expert-like corrected skeleton.

The six technical criteria constrain coaching, but they do not calculate the
displayed total.

## 1. End-to-End Call Graph

```text
source videos
  |
  v
scripts/extract_skeleton_sequences.py
  -> VideoProcessor.process_frames()
  -> PoseDetector.get_pose()
  -> estimate_handedness()
  -> VideoAnalyzer.find_analysis_window()
  -> normalize_skeleton_sequence()
  -> resample_sequence(64)
  -> NPZ sequences
  |
  v
python -m badminton_analysis.ml.train_skeleton_corrector
  -> split experts and students
  -> phase_align_sequence()
  -> nearest training expert
  -> project_bone_lengths()
  -> SkeletonCorrectionPairDataset
  -> SkeletonDenoiser
  -> sequence_training_losses()
  -> checkpoint acceptance gates
  -> .pt checkpoint with expert reference bank
  |
  v
python -m badminton_analysis.ml.infer_skeleton_corrector
  -> predict_correction()
  -> correction_distance()
  -> fit_score_calibration()
  -> phase_grading_details()
  -> keypoint_advice_details()
  -> grades, distances, calibration, advice JSONL
  |
  +--> scripts/render_skeleton_correction_video.py
  |      -> detected and corrected skeleton overlay
  |
  +--> scripts/analyze_clear_with_openai.py
         -> exact clear criteria and checkpoints
         -> correction-distance score context
         -> timestamped zh-TW feedback and joint IDs
         -> renderer pauses and circles target joints
```

## 2. Source Pose Extraction

### Entry point

`scripts/extract_skeleton_sequences.py`

### Runtime prerequisites

The Python project contains PyTorch, OpenCV, OpenAI, and data-processing
dependencies. The pose extractor additionally requires a compatible OpenMMLab
stack (`mmengine`, `mmcv`, `mmdet`, and `mmpose`) with the RTMPose/RTMW model
configs available. Install that stack for the target CUDA/PyTorch environment;
it is intentionally loaded lazily so scoring and unit tests can run on machines
without the GPU pose packages.

### Runtime services

- `badminton_analysis/services/pose_detector.py`
- `badminton_analysis/services/video_processor.py`
- `badminton_analysis/services/video_analyzer.py`

`PoseDetector` uses a whole-body 2D detector and a 3D pose lifter. The processor
retains three aligned streams:

```text
frames                  original video frames
original_landmarks      17-joint COCO 3D poses
body_landmarks_2d       17-joint COCO pixel poses
```

The 2D stream is used for handedness and video rendering. The 3D stream is used
for normalization, training, correction, and scoring.

For a new skill, verify that its source videos produce stable shoulders, hips,
elbows, and wrists before training. Missing data is interpolated for geometry,
but the original confidence mask remains zero so interpolated samples do not
receive normal training or scoring weight.

## 3. Handedness

Implementation: `badminton_analysis/ml/handedness.py`

The estimator does not infer handedness from a single pose. It:

1. interpolates left and right wrist trajectories;
2. subtracts the torso center;
3. scales motion by shoulder width;
4. smooths the trajectories;
5. calculates robust peak wrist speed and positive acceleration;
6. selects a hand only when its motion score is at least twice the other hand.

Fallback order:

```text
motion estimate
  -> datasets/skeleton_sequences/<skill>/handedness_overrides.json
  -> filename metadata
```

Left-handed samples are converted to a dominant-side canonical representation
by swapping every left/right COCO joint pair. In 3D, depth is also mirrored to
preserve chirality. Canonical joint `6` therefore always means dominant shoulder,
but the renderer maps it back to the person's physical right or left side.

This step must remain before analysis-window detection because the window
detector follows the dominant wrist and elbow.

## 4. Analysis Window and Phase Anchors

For clear, extraction calls:

```python
VideoAnalyzer.find_analysis_window(
    skill=Skill.CLEAR,
    hand_positions=dominant_wrist,
    elbow_positions=dominant_elbow,
)
```

The result is `(start, peak, end)`. The extracted sequence stores five anchors:

```text
0: start
1: midpoint(start, peak)
2: peak
3: midpoint(peak, end)
4: end
```

After resampling, clear currently has five sequence-specific phase indices in a
64-frame clip. Model training warps them to:

```text
(0, 16, 32, 48, 63)
```

For another skill, the first required implementation is a reliable
skill-specific analysis window. Do not reuse clear's dominant-wrist peak if the
new skill is defined by foot contact, racket preparation, or another event.

## 5. NPZ Data Contract

Each extracted file under
`datasets/skeleton_sequences/<skill>/{beginners,experts}/` contains:

```text
skeleton_3d       float32 (64, 17, 3)
skeleton_2d       float32 (64, 17, 2)
confidence        float32 (64, 17)
skill             scalar string
handedness        scalar string: left or right
video_name        scalar string
analysis_window   int64 (3,)
phase_indices     int64 (5,)
fps               float32 scalar
```

The generated datasets are runtime artifacts and are not source-controlled.

## 6. Skeleton Normalization

Implementation: `badminton_analysis/ml/skeleton_normalization.py`

`normalize_skeleton_sequence()` performs:

1. temporal interpolation of missing coordinates;
2. dominant-side left/right canonicalization;
3. pelvis centering on every frame;
4. rotation into a fixed body basis from the first valid frame;
5. scaling by median shoulder width;
6. dominant-side depth correction for left-handed 3D poses.

The body basis is fixed for the sequence. This removes camera orientation while
preserving the player's rotation during the stroke. Recomputing the basis every
frame would incorrectly remove the torso rotation that the model must learn.

`phase_align_sequence()` then piecewise-linearly maps the sample's five anchors
onto the canonical phase timeline. `restore_phase_timing()` applies the inverse
mapping to the predicted correction delta.

## 7. Expert-Guided Training Target

Implementation: `badminton_analysis/ml/train_skeleton_corrector.py`

The 50 expert and 50 student clear clips are split independently:

```text
70% train / 10% validation / 20% test
```

With 50 clips, this is 35 training, 5 validation, and 10 untouched test clips.

For each training student:

1. phase-align the student and all training experts;
2. compute confidence-masked mean per-joint Euclidean distance;
3. choose the nearest training expert;
4. adapt that expert to the student's bone lengths with
   `project_bone_lengths()`;
5. use the full adapted expert skeleton as the target.

There is no partial blend between student and expert in v3.

Expert training examples reconstruct their own phase-aligned expert motion.
Student examples learn the full move to the adapted nearest expert.

## 8. Model

Implementation:

- `badminton_analysis/ml/models/skeleton_denoiser.py`
- `badminton_analysis/ml/skeleton_dataset.py`

The accepted model is a separable temporal/spatial Transformer:

```text
frames             64
joints             17
input features     7 per joint
model width        128
attention heads    4
temporal layers    3
spatial layers     2
```

The seven features are:

```text
student XYZ          3
adapted expert XYZ   3
confidence           1
```

The model predicts a correction residual, which is added to the student XYZ.
The result is projected back onto the source skeleton's bone lengths.

Training augmentation adds coordinate noise, masks random joints, shifts time,
and applies coherent limb offsets.

## 9. Training Loss

Implementation: `sequence_training_losses()` in
`badminton_analysis/ml/skeleton_scoring.py`.

```text
training_loss =
    position_error
  + 0.5 * velocity_error
  + 0.25 * normalized_angle_error
  + bone_length_error
```

Position and velocity use the configured joint importance weights. Angle loss
uses eight limb-angle triplets. Bone loss uses the twelve skeleton edges.

The model checkpoint is not accepted merely because validation loss decreases.
The clear acceptance gates require:

```text
all validation students improve toward experts
all corrected validation students enter the expert distance range
mean corrected distance <= validation expert threshold
model-to-selected-reference distance <= 0.10
bounded maximum joint correction
mean correction acceleration <= 0.04
p95 relative bone change < 0.001
```

The accepted clear checkpoint is reference-conditioned and contains its
training-expert reference bank.

## 10. Inference

Implementation: `predict_correction()` in
`badminton_analysis/ml/infer_skeleton_corrector.py`.

Inference repeats the training geometry:

1. phase-align source skeleton and confidence;
2. find the nearest expert in the checkpoint's reference bank;
3. adapt the expert to source bone lengths;
4. concatenate source, reference, and confidence;
5. run the Transformer;
6. apply the full residual;
7. enforce source bone lengths;
8. restore the correction delta to source phase timing;
9. enforce source bone lengths again.

The corrected skeleton is expected to be close to an actual expert reference.
That is separately checked with `expert_euclidean_distances()` and is not implied
by a high calibrated score.

## 11. Correction Distance

The displayed score begins with the difference between the student's original
skeleton and the corrected skeleton.

Implementation: `correction_distance()` in
`badminton_analysis/ml/skeleton_scoring.py`.

```text
D =
    1.00 * position_distance
  + 0.50 * angle_distance
  + 0.50 * velocity_distance
  + 0.25 * bone_length_distance
```

Components:

- `position_distance`: confidence- and joint-weighted mean 3D displacement;
- `angle_distance`: confidence-masked normalized limb-angle change;
- `velocity_distance`: confidence- and joint-weighted motion change;
- `bone_length_distance`: confidence-masked bone-length change.

The current joint weights emphasize the dominant wrist, dominant elbow, and
dominant shoulder. This configuration is suitable for an overhead clear. A
footwork or net skill needs a reviewed weight table rather than blindly reusing
these weights.

## 12. Score Calibration

Implementation: `fit_score_calibration()` and `ScoreCalibration`.

```text
score(D) = 100 * exp(-alpha * max(D - offset, 0))
```

Current clear calibration:

```text
offset = 0.24125837235747438
alpha  = 2.680872933195534
```

The fitter searches expert-distance quantiles for the offset and solves alpha
for the requested student mean. The current target bands are:

```text
students mean = 45
experts mean  = 99
```

Current results:

```text
students: 50 clips, mean 45.00
experts:  50 clips, mean 99.90
```

This is group calibration, not human validation. The student mean is a fitted
target, so it cannot be cited as independent proof of score accuracy. Before
production use, fit a grade mapping on per-video human labels and evaluate MAE,
rank correlation, and calibration on a held-out set.

## 13. Six Criterion Scores

Implementation: `DETAILS` and `phase_grading_details()` in
`badminton_analysis/ml/infer_skeleton_corrector.py`.

Clear maps six correction windows to the existing six criterion maxima:

```text
Preparation correction       10
Rotation correction          10
Balance correction           20
Contact correction           20
Wrist/arm correction         20
Follow-through correction    20
```

Each detail first receives its own calibrated correction-distance score. The six
details are then proportionally reconciled so their sum equals the total score.
They are explanatory allocations of the total, not six independently calibrated
human rubric grades.

`keypoint_advice_details()` also attributes position, angle, velocity, and bone
distance to individual joints and phases. These values feed coaching evidence;
they must not be summed to reconstruct the total.

## 14. OpenAI Coaching Pass

Implementations:

- `badminton_analysis/ml/clear_feedback.py`
- `scripts/analyze_clear_with_openai.py`
- `scripts/render_skeleton_correction_video.py`

The model receives:

- ordered checkpoint-aware images;
- handedness and canonical/physical-side explanation;
- the exact six clear criterion names and technical definitions;
- current expert angle means and standard deviations;
- correction-distance total, components, and six detail scores;
- lowest keypoint correction scores and directions.

The structured response is restricted to:

- `language = zh-TW`;
- one to three exact criterion names;
- an allowed original checkpoint for each criterion;
- canonical coaching target joint IDs;
- Traditional Chinese feedback and visual evidence.

After the response, criterion scores and coaching joint IDs are attached
deterministically from the correction results. The renderer:

1. overlays detected skeleton in cyan;
2. overlays corrected skeleton in green;
3. pauses two seconds at each reported frame;
4. circles the detected joints that require attention;
5. labels the physical dominant shoulder side;
6. shows the correction-distance total and criterion allocation.

The API key is loaded from `.env` and is never written to prompt artifacts.

## 15. Generated Artifacts

Core inference writes:

```text
grading_results.csv       full rows and six detail scores
all_grades.csv            compact score export
distance_components.csv   raw distance components and quality metrics
score_summary.csv         group means and separation
calibration.json          offset and alpha
keypoint_scores.csv       per-joint/per-phase evidence
advice_context.jsonl      ranked language-model context
```

Coaching writes:

```text
input_frames/
prompt_context.json
feedback.json
feedback.csv
annotated_feedback_h264.mp4
```

Generated datasets, statistics, and videos are intentionally not committed to
Git. The accepted v3 checkpoint and its calibration are committed so a clean
checkout can run inference.

## 16. Reproducing Clear

```bash
.venv/bin/python scripts/extract_skeleton_sequences.py

.venv/bin/python -m badminton_analysis.ml.train_skeleton_corrector \
  --dataset-root datasets/skeleton_sequences/clear \
  --model-path models/skeleton_correction/clear_expert_guided_v3.pt

.venv/bin/python -m badminton_analysis.ml.infer_skeleton_corrector \
  --dataset-root datasets/skeleton_sequences/clear \
  --model-path models/skeleton_correction/clear_expert_guided_v3.pt \
  --output-dir stats/skeleton_correction/clear_expert_guided_v3_grades

.venv/bin/python scripts/export_clear_correction_scores.py

.venv/bin/python scripts/analyze_clear_with_openai.py \
  --video-path stats/skeleton_correction/clear_expert_guided_v3_videos/students/EG3.mp4 \
  --dataset-path datasets/skeleton_sequences/clear/beginners/EG3.npz \
  --output-dir stats/openai_clear_feedback/EG3
```

## 17. Adding Another Skill

Do not copy the clear module and only change directory names. Perform these
steps explicitly:

1. **Define the skill phase contract.**
   Specify start, peak/contact, end, and five monotonic anchors.

2. **Implement analysis-window detection.**
   Choose the joints and events that define the skill. Footwork should not use
   the clear wrist peak.

3. **Review handedness relevance.**
   Decide whether dominant-side canonicalization is appropriate. Some footwork
   skills may need stroke direction or court side instead.

4. **Create isolated dataset paths.**
   Use `datasets/skeleton_sequences/<skill>/...`; never mix expert banks across
   skills.

5. **Define joint and phase importance.**
   Replace clear's upper-body-heavy `JOINT_WEIGHTS`, `DETAILS`, and
   `KEYPOINT_PHASES` with reviewed skill-specific values.

6. **Define coaching criteria.**
   Add exact criterion names, allowed frames, measured joints, coaching target
   joints, and Traditional Chinese wording.

7. **Train a separate checkpoint.**
   Use only that skill's experts and preserve independent train, validation, and
   untouched test splits.

8. **Measure natural expert variability.**
   Use leave-one-out expert distance to define the geometry acceptance range.

9. **Run geometry gates before score calibration.**
   Verify every held-out correction moves toward experts and visually inspect
   low, middle, and high corrections.

10. **Fit a skill-specific calibration.**
    Never reuse clear's offset or alpha. Prefer human per-video grades; group
    target fitting is diagnostic only.

11. **Validate criterion attribution.**
    Ensure every displayed criterion score is tied to the intended phase and
    joints and that all six allocations sum to the total.

12. **Validate the coaching video.**
    Check physical handedness, circled joints, pause frame, Traditional Chinese
    text, and overlay alignment.

## 18. Recommended Refactor Before Many Skills

Clear-specific constants are still distributed across scripts. Before the
second or third skill, introduce a `SkillCorrectionSpec` containing:

```text
skill identifier
source video groups
analysis-window callback
canonical phase anchors
joint weights
detail phase windows and maxima
criterion names and rule references
allowed feedback frames
coaching target joints
dataset/checkpoint/output paths
```

Then make extraction, training, inference, score export, and coaching consume the
spec. Keep the model and scoring primitives skill-agnostic. This avoids a family
of nearly identical scripts with silently different scoring behavior.
