# Multi-Skill Skeleton-Correction Pipeline

This document explains the knowledge needed to understand the code, the full
architecture and grading algorithm, and the recommended code-tracing order.

The supported correction skills are serve, lift, clear, and smash. They share
an engine but remain separate experiments:

```text
datasets/skeleton_sequences/<skill>/
models/skeleton_correction/<skill>_expert_guided_<version>.pt
models/skeleton_correction/<skill>_expert_guided_<version>.calibration.json
stats/skeleton_correction/<skill>_expert_guided_<version>_*/
```

No expert bank, checkpoint, or calibration is shared across skills. The clear
checkpoint is currently committed. Serve, lift, and smash require their own
data extraction, training, acceptance audit, and calibration runs.

## 1. Knowledge Needed To Read The Code

### Python And Data Structures

You should be comfortable with:

- Python dataclasses, enums, type annotations, mappings, and command-line
  parsers;
- NumPy broadcasting, masking, interpolation, norms, and array shapes;
- PyTorch modules, tensors, optimizers, data loaders, checkpoints, and
  inference mode;
- pandas CSV operations; and
- Pydantic models and validators for structured LLM output.

The main tensor shapes are:

```text
one pose sequence       (T, J, 3)
confidence              (T, J)
expert bank             (N, T, J, 3)
model input             (B, T, J, 7)
model output            (B, T, J, 3)

T = 64 frames
J = 17 COCO joints
```

### Pose And Coordinate Concepts

You should know the 17-joint COCO skeleton and the distinction between:

- image coordinates: pixels in the original camera view;
- lifted 3D coordinates: the pose lifter's estimated body geometry;
- body-normalized coordinates: pelvis-centered coordinates scaled by the median
  observed anatomical-segment length;
- physical left/right joints; and
- canonical dominant/non-dominant joints.

After canonicalization, joint `6` is always the dominant shoulder, joint `8`
the dominant elbow, and joint `10` the dominant wrist. For a left-handed player,
those canonical IDs still map to the physical left side when rendered.

### Time And Phase Concepts

Three timelines must not be confused:

```text
original video frames
  -> retained pose frames
  -> normalized 64-frame model sequence
```

The NPZ stores mappings back to the original video. Phase alignment is a
piecewise-linear temporal warp: it aligns five sample-specific anchors to one
canonical five-anchor timeline before expert comparison.

### Machine-Learning Concepts

The model is supervised by pseudo-targets, not manually corrected skeletons.
For each student, the target is the nearest phase-aligned training expert after
adapting that expert pose to the student's bone lengths. Understand:

- confidence-masked, skill-weighted correction distance;
- nearest-neighbor reference selection;
- residual prediction;
- temporal and spatial self-attention;
- multi-component geometry losses;
- train/validation/test separation; and
- calibration as a separate operation from model training.

### Evaluation Concepts

A high score means the model predicts that little correction is required. It
does not independently prove that the movement is correct. The current
calibration deliberately targets a student group mean near 45 and an expert
group mean near 99. Human per-video grades are still required for independent
validity testing.

## 2. End-To-End Architecture

```text
videos for exactly one skill
  |
  v
scripts/extract_skeleton_sequences.py --skill <skill>
  -> VideoProcessor.process_frames()
  -> PoseDetector.get_pose()
  -> direct RTMW3D whole-body 2D/3D pose
  -> original-frame provenance
  -> estimate_handedness()
  -> VideoAnalyzer.find_analysis_window(skill=...)
  -> normalize_skeleton_sequence()
  -> resample_sequence(64)
  -> phase_indices + source_frame_indices
  -> skill-specific NPZ dataset
  |
  v
python -m badminton_analysis.ml.train_skeleton_corrector --skill <skill>
  -> split experts and students independently
  -> phase_align_sequence()
  -> project_bone_lengths() for each training expert
  -> skill-weighted nearest adapted expert
  -> SkeletonCorrectionPairDataset
  -> SkeletonDenoiser
  -> skill-weighted training losses
  -> held-out expert-distance and geometry acceptance gates
  -> skill-specific checkpoint and expert bank
  |
  v
python -m badminton_analysis.ml.infer_skeleton_corrector --skill <skill>
  -> predict_correction()
  -> bone-length projection
  -> skill-weighted correction_distance()
  -> fit_score_calibration()
  -> skill-specific criterion allocations
  -> per-phase/per-joint evidence
  -> CSV, JSONL, and calibration outputs
  |
  +--> scripts/render_skeleton_correction_video.py
  |      -> detected and corrected skeletons on one video
  |
  +--> scripts/render_all_skeleton_correction_videos.py --skill <skill>
  |      -> all student and expert videos with scores
  |
  +--> scripts/analyze_skill_with_openai.py --skill <skill>
         -> exact skill criteria and allowed anchors
         -> correction score and joint evidence
         -> validated Traditional Chinese feedback
         -> timestamped joint IDs for pauses and circles
```

## 3. Skill Isolation And Specifications

Start at:

```text
badminton_analysis/ml/skill_specs.py
```

`SkillCorrectionSpec` is the contract consumed by extraction, training,
inference, scoring, export, feedback, and rendering. Each skill defines:

```text
skill enum and names
checkpoint roles
17 joint weights
criterion correction windows
phase evidence windows
Traditional Chinese qualitative rules
measured joints
coaching-circle joints
allowed grading anchors
dataset/model/output defaults
```

The four specifications are independent objects. A checkpoint also stores its
skill, joint weights, and criteria. Inference and rendering reject a checkpoint
whose skill does not match the requested skill or dataset.

### Serve Criteria

```text
10  雙手平舉
10  將重心放至持拍腳
20  重心轉移至非持拍腳
20  髖關節前旋
20  持拍手手腕發力
20  肩膀旋轉朝前
```

Serve weights emphasize the pelvis, knees, ankles, dominant arm, and shoulder
because weight transfer and body rotation are part of the qualitative contract.

### Lift Criteria

```text
10  手腕放置腰部放鬆預備
25  手腕往後引拍
35  手腕往前壓
30  手腕放鬆回到預備姿勢
```

Lift weights emphasize the dominant shoulder, elbow, and wrist. The final
criterion explicitly evaluates relaxation and return to the ready position.

### Clear Criteria

```text
10  球拍舉至腰部預備
10  轉身
20  雙手手肘平衡
20  手肘往前轉至前方
20  手腕發力
20  慣用手肩膀往前轉
```

### Smash Criteria

Smash retains the same six qualitative names and maxima as clear. It is still a
separate skill: its joint weights give greater importance to the dominant arm,
shoulder, hips, and explosive follow-through, and it must be trained and
calibrated only on smash data.

## 4. Pose Extraction And Frame Provenance

Entry point:

```text
scripts/extract_skeleton_sequences.py
```

Runtime services:

```text
badminton_analysis/services/video_processor.py
badminton_analysis/services/pose_detector.py
badminton_analysis/services/video_analyzer.py
```

`VideoProcessor` does not grade. It returns aligned retained-frame streams:

```text
frames                    retained RGB frames
original_landmarks        COCO 3D pose dictionaries
body_landmarks_2d         COCO 2D pose dictionaries
wholebody_landmarks       whole-body 2D dictionaries
source_frame_indices      original frame ID for every retained pose frame
```

Frames without a usable pose can still be omitted by the current processor,
but the time axis is no longer anonymous: retained frames keep their original
frame IDs.

The extracted NPZ contains:

```text
skeleton_3d               (64, 17, 3)
skeleton_2d               (64, 17, 2)
confidence                (64, 17)
skill                     scalar string
handedness                scalar string
video_name                scalar string
analysis_window           start, peak, end in retained-pose space
phase_indices             five anchors in normalized 0..63 space
source_frame_indices      original frame ID for every normalized frame
source_phase_indices      original frame IDs for the five anchors
fps                       original video FPS
```

`phase_indices` must be used for model tensors and rendered 64-frame correction
videos. `source_frame_indices` must be used when sampling the original video.

## 5. Handedness

Implementation:

```text
badminton_analysis/ml/handedness.py
```

The detector:

1. interpolates both wrist tracks where possible;
2. normalizes movement by torso scale;
3. computes wrist acceleration magnitudes;
4. derives a robust motion score for each side; and
5. accepts a side only when its score is at least twice the other side.

If the evidence is ambiguous, extraction uses an explicit metadata override or
the filename fallback. The output records the selected source and confidence
ratio.

## 6. Skill-Specific Analysis Windows

Implementation:

```text
VideoAnalyzer.find_analysis_window()
```

The current structural parser uses dominant wrist and elbow motion:

- clear and smash use the overhead/smash window heuristic;
- serve uses the serve window heuristic; and
- lift uses the low backswing followed by the highest-hand completion.

Every window becomes five anchors:

```text
start
midpoint(start, peak)
peak
midpoint(peak, end)
end
```

The meaning of those anchors is supplied by the selected skill specification.
This code does not yet track the shuttle or racket, so a motion peak is not a
verified shuttle-contact event. Adding hit-centric shuttle/racket parsing is the
next structural improvement, but is outside the current correction replication.

## 7. Skeleton Normalization

Implementation:

```text
badminton_analysis/ml/skeleton_normalization.py
```

Normalization performs:

1. interpolation of missing coordinates where evidence exists;
2. left/right swapping for left-handed players;
3. depth mirroring to preserve chirality;
4. pelvis centering;
5. alignment to a body basis derived from shoulders and torso;
6. shoulder-width scaling; and
7. resampling to 64 frames.

Camera translation, body size, and handedness are reduced while relative joint
geometry and motion remain available.

`phase_align_sequence()` then maps the sample's five anchors to the canonical
timeline. `restore_phase_timing()` maps predicted correction deltas back to the
sample's timing.

## 8. Expert Pairing And Pseudo-Targets

Implementation:

```text
badminton_analysis/ml/train_skeleton_corrector.py
badminton_analysis/ml/skeleton_scoring.py
```

For source sequence `S`, expert sequence `E_n`, source confidence `C`, and
expert confidence `C_n`, each reference is first adapted to the student:

```text
M_n(t,j) = C(t,j) * C_n(t,j)
A_n = project_bone_lengths(S, E_n)

d_n = correction_distance(S, A_n, M_n, skill_joint_weights)

nearest = argmin_n d_n
```

The distance is the same weighted position, angle, velocity, and bone metric
used for grading after phase alignment. It is not cosine similarity or a
separate embedding score. `project_bone_lengths()` keeps the student's pelvis
anchor, and the selected complete adapted expert motion is the pseudo-target.

Expert training samples reconstruct their own expert movement. Student samples
learn the full movement toward their adapted nearest expert. Only experts from
that skill's training split can enter the checkpoint reference bank.

## 9. Model Input And Architecture

Implementations:

```text
badminton_analysis/ml/skeleton_dataset.py
badminton_analysis/ml/models/skeleton_denoiser.py
```

The reference-conditioned input has seven features per joint:

```text
[source_x, source_y, source_z,
 reference_x, reference_y, reference_z,
 confidence]
```

The model:

1. projects the seven values into the model dimension;
2. adds learned joint and time embeddings;
3. applies temporal Transformer layers independently along each joint track;
4. applies spatial Transformer layers across all joints in each frame;
5. predicts a bounded 3D residual; and
6. adds the residual to the source coordinates.

For newly trained checkpoints, inference then blends the learned output with
the adapted reference before bone projection:

```text
guided = 0.50 * model_output + 0.50 * adapted_reference
```

The blend weight is saved as `reference_guidance` in the checkpoint. Legacy
checkpoints without that field default to zero. Inference finally projects the
result back onto the student's bone lengths.

## 10. Training Loss

Implementation:

```text
sequence_training_losses()
```

Let `P` be the prediction, `T` the adapted expert target, `M` confidence, and
`w_j` the selected skill's joint weights.

Position loss is a weighted masked mean:

```text
L_position = sum(M * w * ||P - T||_2) / sum(M * w)
```

Velocity loss applies the same calculation to consecutive-frame differences.
Angle loss compares eight limb/torso angle triplets and normalizes radians by
pi. Bone loss compares the lengths of the 12 skeleton edges.

```text
L_train = L_position
        + 0.50 * L_velocity
        + 0.25 * L_angle
        + 1.00 * L_bone
```

Training augmentation adds coordinate noise, masks joints, shifts time, and
adds coherent limb offsets.

## 11. Checkpoint Acceptance

A lower validation loss alone does not save a checkpoint. The trainer also
evaluates corrected validation students. Reference selection is restricted to
training experts; correctness distances are reported against the union of
those permitted experts and unseen held-out experts, plus against the unseen
experts alone.

The default acceptance gates require:

```text
improved student fraction                 >= 1.0
within held-out expert range fraction     >= 1.0
corrected distance                        <= held-out expert p95 range
reference-to-target distance              <= 0.1
maximum joint correction                  <= 1.5 * configured bound
mean correction/reference acceleration    <= 1.10
p95 relative bone-length change           < 0.001
```

The acceleration gate compares the predicted correction with the full
bone-adapted expert correction required for the same input. It rejects added
jitter without treating the naturally faster clear and smash motions as
failures. After training, the accepted checkpoint is audited again on the test
split and all students. Each skill must independently pass these gates.

## 12. Inference And Corrected Skeleton Verification

Implementation:

```text
badminton_analysis/ml/infer_skeleton_corrector.py
```

Inference:

1. validates the checkpoint skill;
2. phase-aligns the source;
3. adapts every checkpoint training expert to source bone lengths;
4. selects the adapted expert with the lowest skill-weighted grading distance;
5. constructs seven-feature input;
6. predicts the residual correction;
7. blends the learned output toward the adapted reference using the saved
   `reference_guidance`;
8. projects bone lengths;
9. restores source timing; and
10. projects bone lengths again.

Training and checkpoint acceptance explicitly measure the Euclidean distance
between corrected students and experts. The primary acceptance metric uses the
training-plus-unseen union so it can verify the selected reference directly;
`corrected_nearest_unseen_expert_distance` is retained as the stricter
generalization diagnostic. Both are separate from the displayed
correction-distance score.

## 13. Grading Algorithm

Implementation:

```text
badminton_analysis/ml/skeleton_scoring.py
badminton_analysis/ml/infer_skeleton_corrector.py
```

The displayed score measures the change from original skeleton `S` to corrected
skeleton `C`, not the raw distance from the student to one unadapted expert.

For the selected skill's joint weights:

```text
D_position = weighted masked mean(||S - C||_2)
D_velocity = weighted masked mean(||delta(S) - delta(C)||_2)
D_angle    = masked mean absolute angle change / pi
D_bone     = masked mean absolute bone-length change

D = D_position
  + 0.50 * D_angle
  + 0.50 * D_velocity
  + 0.25 * D_bone
```

The score transform is:

```text
score(D) = 100 * exp(-alpha * max(D - offset, 0))
```

`fit_score_calibration()` is run separately for each skill. It searches offsets
from the complete evaluated expert cohort's correction-distance distribution
and solves `alpha` so that the complete beginner cohort mean approaches `45.0`
while the complete expert cohort mean approaches `99.8`.

This is group-fitted diagnostic calibration, not an independent test result.
The output marks it as `diagnostic_group_calibrated`. Split labels remain in
the CSV for traceability, but split score means are also affected by the
full-cohort calibration. Use the held-out Euclidean audits above, not those
score means, to judge correction-model generalization. A model cannot reuse the
clear offset or alpha for serve, lift, or smash.

### Criterion Allocations

Each `CorrectionDetailSpec` selects a skill-specific time window and joint set.
It calculates a local correction distance and maps it through the same skill
calibration:

```text
raw criterion grade_k = maximum_k * score(distance_k) / 100
```

The raw criterion grades are proportionally reconciled so that:

```text
sum(criterion grades) = total score
```

The criteria therefore explain where the correction is concentrated. They are
not six or four independent rule graders, and the LLM is not allowed to invent
or modify their numeric values.

## 14. Score And Evidence Outputs

Inference writes:

```text
grading_results.csv       full rows and criterion allocations
all_grades.csv            compact score list
score_summary.csv         group statistics and separation
distance_components.csv   total distance components
keypoint_scores.csv       joint and phase evidence
advice_context.jsonl      structured context for coaching
calibration.json          fitted score parameters
```

`keypoint_advice_details()` assigns each joint:

- total correction distance and score;
- position, angle, velocity, and bone components;
- skill-specific importance weight;
- worst skill phase;
- correction direction and vector; and
- per-phase scores and distances.

## 15. Traditional Chinese LLM Coaching

Implementations:

```text
badminton_analysis/ml/clear_feedback.py
scripts/analyze_skill_with_openai.py
```

Despite the historical filename `clear_feedback.py`, the module now validates
all four skill contracts. It keeps clear-compatible exports for the committed
pipeline.

The prompt contains:

- exactly one selected skill;
- that skill's exact Traditional Chinese criteria;
- permitted normalized grading anchors per criterion;
- measured and coaching-circle joint IDs;
- handedness and physical-side explanation;
- correction total, components, and allocations;
- lowest joint scores and correction directions; and
- ordered high-detail images.

Pydantic and post-response validation reject:

- a different or unsupported skill;
- an unknown or cross-skill criterion;
- a title that does not exactly match its rule ID;
- a phase that does not match the criterion;
- joints outside the measured criterion contract;
- a frame outside the supplied images;
- a frame outside the criterion's allowed anchors; and
- non-Chinese feedback text.

After validation, coaching joint IDs are deterministically replaced with the
configured physical coaching targets. The LLM cannot select the wrong shoulder
for a left-handed player.

`--video-frame-space normalized` samples a generated 64-frame correction video.
`--video-frame-space source` uses the stored original-frame mapping when
sampling the raw source video.

## 16. Rendering

Implementations:

```text
scripts/render_skeleton_correction_video.py
scripts/render_all_skeleton_correction_videos.py
scripts/render_skeleton_correction_overlays.py
```

The renderer:

1. re-extracts pixel skeletons from the source video;
2. resamples them to the model's 64-frame timeline;
3. fits the corrected 3D pose to the detected 2D pose;
4. overlays detected and corrected skeletons;
5. displays the calibrated score beside the name;
6. maps canonical dominant joints back to the physical body side;
7. pauses at validated feedback frames;
8. circles deterministic coaching joints; and
9. renders concise Traditional Chinese advice.

Generated video outputs remain ignored by Git.

## 17. Recommended Code-Tracing Order

Read the files in this order.

### Pass 1: Contracts And Entry Points

1. `badminton_analysis/models/types.py`
2. `badminton_analysis/ml/skill_specs.py`
3. `scripts/extract_skeleton_sequences.py`
4. `badminton_analysis/ml/train_skeleton_corrector.py:main()`
5. `badminton_analysis/ml/infer_skeleton_corrector.py:main()`

At the end of this pass, you should know which skill is selected, where its
artifacts live, and the high-level call graph.

### Pass 2: Video To Model Tensor

1. `badminton_analysis/services/video_processor.py`
2. `badminton_analysis/services/pose_detector.py`
3. `badminton_analysis/ml/handedness.py`
4. `badminton_analysis/services/video_analyzer.py`
5. `badminton_analysis/ml/skeleton_normalization.py`
6. `badminton_analysis/ml/skeleton_dataset.py`

Track these values on paper:

```text
original frame ID
retained pose frame ID
analysis start/peak/end
normalized frame ID
five phase anchors
source-frame mapping
handedness canonicalization
```

### Pass 3: Target Construction And Training

1. `_load_aligned()`
2. `_load_expert_bank()`
3. `_build_student_targets()`
4. `select_bone_adapted_expert()`
5. `project_bone_lengths()` and `correction_distance()`
6. `SkeletonCorrectionPairDataset.__getitem__()`
7. `SkeletonDenoiser.forward()`
8. `sequence_training_losses()`
9. `_evaluate_expert_distance()`
10. the checkpoint acceptance block in `train_skeleton_corrector.py`

Verify that only the selected skill's training experts enter the reference bank.

### Pass 4: Inference And Grades

1. `load_corrector()`
2. `predict_correction()`
3. `correction_distance_components()`
4. `correction_distance()`
5. `fit_score_calibration()`
6. `phase_grading_details()`
7. `keypoint_advice_details()`
8. `SkeletonCorrectionBackend.score()`
9. `badminton_analysis/tools/grade_students.py`

Calculate one sample manually through `D` and the exponential score transform
before reading the CSV export code.

### Pass 5: Coaching And Video

1. `badminton_analysis/ml/clear_feedback.py`
2. `scripts/analyze_skill_with_openai.py`
3. `scripts/render_skeleton_correction_video.py`
4. `scripts/render_all_skeleton_correction_videos.py`

Confirm that criterion names, frames, phases, and circle joints all come from
the selected `SkillCorrectionSpec`.

### Pass 6: Tests

Read:

```text
tests/test_skill_specs.py
tests/test_skeleton_correction.py
tests/test_clear_feedback.py
tests/test_grade_students.py
tests/test_video_processor.py
```

The tests provide small executable examples of every important contract.

## 18. Reproducing One Skill

Run each skill separately. For example, lift:

```bash
.venv/bin/python scripts/extract_skeleton_sequences.py \
  --skill lift \
  --beginner-dir /path/to/lift/students \
  --expert-dir /path/to/lift/experts

.venv/bin/python -m badminton_analysis.ml.train_skeleton_corrector \
  --skill lift

.venv/bin/python -m badminton_analysis.ml.infer_skeleton_corrector \
  --skill lift

.venv/bin/python scripts/export_correction_scores.py \
  --skill lift

.venv/bin/python scripts/render_all_skeleton_correction_videos.py \
  --skill lift \
  --student-video-dir /path/to/lift/students \
  --expert-video-dir /path/to/lift/experts
```

Then inspect, in order:

```text
dataset summary
training pair CSV
expert variability JSON
validation acceptance metrics
test expert-distance audit
grading results
group score summary
keypoint evidence
rendered skeleton videos
LLM prompt context and feedback
```

Do not copy a calibration file from another skill, do not mix experts across
skills, and do not treat a passed group-separation target as human validation.
