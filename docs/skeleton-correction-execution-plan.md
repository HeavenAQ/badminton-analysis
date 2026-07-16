# Skeleton-Correction Scoring Execution Plan

## Objective

Replace the current rule/prototype scoring path with a skeleton-correction score:

```text
student skeleton sequence
-> model predicts corrected expert-like skeleton
-> score = weighted distance(student skeleton, corrected skeleton)
```

Start with badminton clear only. Prove that the score distribution is feasible before implementing every skill.

Feasibility means:

- Expert clear videos should score around 98-100 on average.
- Beginner/student clear videos should score around 40-50 on average.
- Beginner/student clear videos receive a meaningfully lower average score than the current prototype scorer.
- The correction overlays are physically plausible and explain why a score is low.
- The method handles handedness and analysis-window alignment correctly.

Do not begin with pixel-level video generation. The model should operate on skeleton sequences.

## Repo Context

Important existing files:

- `badminton_analysis/tools/grade_students.py`
- `badminton_analysis/services/video_processor.py`
- `badminton_analysis/services/video_analyzer.py`
- `badminton_analysis/services/body_normalizer.py`
- `badminton_analysis/services/graders/base.py`
- `badminton_analysis/services/graders/clear.py`
- `scripts/train_prototype_scorer.py`
- `stats/clear/prototype_scorer.json`

The current grading CLI is:

```bash
.venv/bin/python -m badminton_analysis.tools.grade_students \
  --skill clear \
  --input-dir scoring_videos/高遠球/初學者高遠球 \
  --output-dir stats/some_output_dir
```

The existing pipeline extracts pose with `VideoProcessor.process_frames()`, finds the analysis window with `VideoAnalyzer.find_analysis_window()`, selects five key frames, and scores those checkpoints through a skill grader.

The new architecture should reuse the same pose extraction and analysis-window logic at first. Avoid changing window detection until the scoring approach is validated.

## Remote Execution

Use the remote `nislab` machine for model/data runs. The user has explicitly allowed agents to use `nislab` for scoring and training when needed.

Sync local source to the remote repo:

```bash
rsync -avP ./ nislab:~/heaven/badminton-analysis/ \
  --exclude .git \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude .pytest_cache \
  --exclude annotation-frontend/.nuxt \
  --exclude annotation-frontend/.output \
  --exclude annotation-frontend/dist \
  --exclude annotation-frontend/public/annotation-images
```

Run commands on the remote:

```bash
ssh nislab
cd ~/heaven/badminton-analysis
```

Copy generated outputs back:

```bash
rsync -avP nislab:~/heaven/badminton-analysis/stats/skeleton_correction/ \
  stats/skeleton_correction/
```

## Stage 1: Establish Clear Baselines

Run current scorer on clear beginners and clear experts.

```bash
.venv/bin/python -m badminton_analysis.tools.grade_students \
  --skill clear \
  --input-dir scoring_videos/高遠球/初學者高遠球 \
  --output-dir stats/skeleton_correction/baseline_clear_beginners

.venv/bin/python -m badminton_analysis.tools.grade_students \
  --skill clear \
  --input-dir scoring_videos/高遠球/專家高遠球 \
  --output-dir stats/skeleton_correction/baseline_clear_experts
```

Deliverables:

- `stats/skeleton_correction/baseline_clear_beginners/grading_results.csv`
- `stats/skeleton_correction/baseline_clear_experts/grading_results.csv`
- A short baseline summary with beginner mean, expert mean, median, min, max, and overlap.

Acceptance:

- Both runs complete.
- Failures are investigated.
- The baseline score distribution is documented before new-model work begins.

## Stage 2: Build Skeleton Sequence Dataset

Add:

```text
scripts/extract_skeleton_sequences.py
```

Target output:

```text
datasets/skeleton_sequences/clear/beginners/{video_id}.npz
datasets/skeleton_sequences/clear/experts/{video_id}.npz
```

Each `.npz` should contain:

```text
skeleton_3d: T x J x 3
skeleton_2d: T x J x 2
confidence: T x J
wholebody_2d: optional
skill: clear
handedness: right/left
video_name
analysis_window: start, peak, end
phase_indices: five key-frame indices
fps or time_intervals when available
```

Use existing code paths:

- `VideoProcessor.process_frames()`
- `VideoAnalyzer.find_analysis_window()`
- `VideoAnalyzer.compute_angles()`
- existing handedness conventions in `Handedness`

Normalize every sequence:

```text
1. Mirror left-handed players into dominant-side coordinates.
2. Root-center at pelvis or hip midpoint.
3. Scale by shoulder width or torso length.
4. Optionally rotate into body-centric coordinates using existing body normalizer.
5. Crop to the analysis window.
6. Resample the window to fixed T, initially T = 64.
7. Preserve a confidence mask.
```

Deliverables:

- extraction script
- dataset files for clear beginners and experts
- `stats/skeleton_correction/clear_dataset_summary.csv`

Dataset summary columns:

```text
filename
label
handedness
raw_tracked_frames
analysis_start
analysis_peak
analysis_end
resampled_frames
missing_joint_ratio
status
error
```

Acceptance:

- At least 90% of clear videos produce usable skeleton sequences.
- Left-handed videos are either correctly mirrored or explicitly listed as unsupported.
- Bad pose-tracking cases are visible in the summary.

## Stage 3: Implement First Feasibility Model

Do not start with diffusion. First implement a denoising skeleton Transformer.

Suggested layout:

```text
badminton_analysis/ml/
  __init__.py
  skeleton_dataset.py
  skeleton_normalization.py
  skeleton_scoring.py
  train_skeleton_corrector.py
  infer_skeleton_corrector.py
  models/
    __init__.py
    skeleton_denoiser.py
```

Model:

```text
Input: T x J x F
F initially: x, y, z, confidence
Backbone: temporal Transformer with joint embedding
Decoder: corrected skeleton sequence
```

Training:

```text
input: corrupted expert skeleton
target: clean expert skeleton
```

Corruptions:

- Gaussian joint noise
- random joint masking
- temporal jitter
- small phase/time shifts
- limb-angle perturbations

Loss:

```text
L = position_loss
  + velocity_loss
  + angle_loss
  + bone_length_loss
```

Use weighted joints for badminton clear:

- dominant shoulder
- dominant elbow
- dominant wrist
- non-dominant shoulder
- hips
- knees
- ankles

Deliverables:

- trainable model
- saved checkpoint, e.g. `models/skeleton_correction/clear_denoiser.pt`
- training metrics under `stats/skeleton_correction/clear_training/`

Acceptance:

- Training completes on `nislab`.
- Model reconstructs clean expert sequences from corrupted expert sequences.
- Validation reconstruction loss is stable enough to proceed to scoring.

## Stage 4: Score By Correction Distance

At inference:

```text
student skeleton -> model -> corrected expert-like skeleton
```

Compute correction distance:

```text
distance = w_pos * MPJPE
         + w_angle * joint_angle_error
         + w_vel * velocity_error
         + w_bone * bone_length_error
```

Initial score mapping:

```text
score = 100 * exp(-alpha * normalized_distance)
```

Calibrate `alpha` using clear expert and beginner distributions.

Target distribution:

```text
clear experts mean: 98-100
clear beginners/students mean: 40-50
```

The mapping should be calibrated to this distribution without hiding model failures. If expert correction distances are not near zero, fix normalization, phase alignment, or the correction model before forcing the score mapping. If beginners still score too high, inspect correction overlays and distance components before increasing the penalty.

Produce both total and detail scores. Details can initially be phase-based:

```text
detail_1: preparation correction distance
detail_2: rotation correction distance
detail_3: balance correction distance
detail_4: contact correction distance
detail_5: wrist/arm correction distance
detail_6: follow-through correction distance
```

Deliverables:

```text
stats/skeleton_correction/clear_feasibility/grading_results.csv
stats/skeleton_correction/clear_feasibility/score_summary.csv
stats/skeleton_correction/clear_feasibility/distance_components.csv
stats/skeleton_correction/clear_feasibility/old_vs_new_scores.csv
```

Acceptance:

- Expert mean is around 98-100.
- Beginner/student mean is around 40-50.
- Beginner/student mean is meaningfully lower than the old prototype scorer.
- The method does not simply assign everyone low scores.
- Distance components identify which body parts or phases caused score loss.

## Stage 5: Visual Debugging

Add an overlay/debug script:

```text
scripts/render_skeleton_correction_overlays.py
```

Generate side-by-side or overlaid frames:

```text
original student skeleton
corrected skeleton
correction vectors
per-joint correction magnitude
```

Output:

```text
stats/skeleton_correction/clear_debug_overlays/
```

Create overlays for:

- lowest-scoring 10 beginners
- highest-scoring 10 beginners
- lowest-scoring 10 experts
- random sample of 10

Acceptance:

- Low scores correspond to visible and technically plausible corrections.
- Corrected skeletons do not show obvious limb stretching, joint swapping, or temporal jitter.
- Handedness and dominant-side normalization are visually correct.

## Stage 6: Integrate With Grading CLI

After clear feasibility passes, add a scorer backend option:

```bash
.venv/bin/python -m badminton_analysis.tools.grade_students \
  --skill clear \
  --input-dir scoring_videos/高遠球/初學者高遠球 \
  --output-dir stats/skeleton_correction/clear_cli_test \
  --scorer skeleton-correction \
  --model-path models/skeleton_correction/clear_denoiser.pt
```

Keep existing behavior as default. Supported scorer modes should become:

```text
rule
prototype
skeleton-correction
```

The new backend should write the same core CSV columns:

```text
filename
skill
handedness
status
error
total_grade
start_frame
peak_frame
end_frame
detail_1_desc
detail_1_grade
...
```

Append diagnostic columns:

```text
correction_distance
position_distance
angle_distance
velocity_distance
bone_length_distance
model_path
scorer
```

Acceptance:

- Existing CLI behavior still works without `--scorer`.
- New scorer works for clear.
- CSV schema remains compatible with existing result review scripts.

## Stage 7: Decide Whether To Upgrade Architecture

Only consider a stronger architecture after the denoising Transformer is evaluated.

Upgrade path:

```text
1. Denoising Transformer autoencoder
2. MotionAGFormer-style local/global graph Transformer encoder
3. Masked skeleton inpainting
4. SkeletonDiffusion-style latent diffusion
```

Use diffusion only if:

- Denoising Transformer corrections are too smooth or collapse toward an average pose.
- Temporal motion quality is poor.
- Correction overlays show physically plausible direction but poor sequence realism.

Do not use pixel/video generation unless skeleton-only methods fail and there is a specific reason to model appearance.

## Stage 8: Extend To Other Skills

Implement all skills only after clear passes the feasibility gate.

Recommended order:

```text
clear
smash
serve
lift
drive/net kill
footwork last
```

For each skill:

```text
1. Extract skeleton dataset.
2. Verify analysis-window and phase alignment.
3. Train skill-specific corrector.
4. Evaluate beginner/expert separation.
5. Generate correction overlays.
6. Compare against old scorer.
7. Integrate into CLI.
```

Do not train a shared multi-skill model until each single-skill model is validated.

## Required Tests

Add focused tests where possible:

- normalization preserves shape and mirrors handedness consistently
- resampling returns fixed sequence length
- scoring distance is zero or near-zero for identical sequences
- score decreases as synthetic corruption increases
- CLI default path still produces current grading CSV
- CLI skeleton-correction path writes compatible CSV columns

Run:

```bash
uv run -m pytest -q
```

If remote dependencies require the remote environment:

```bash
ssh nislab
cd ~/heaven/badminton-analysis
.venv/bin/python -m pytest -q
```

## Final Deliverables

The first completed milestone should include:

```text
datasets/skeleton_sequences/clear/
models/skeleton_correction/clear_denoiser.pt
stats/skeleton_correction/clear_feasibility/grading_results.csv
stats/skeleton_correction/clear_feasibility/score_summary.csv
stats/skeleton_correction/clear_feasibility/old_vs_new_scores.csv
stats/skeleton_correction/clear_debug_overlays/
badminton_analysis/ml/
scripts/extract_skeleton_sequences.py
scripts/render_skeleton_correction_overlays.py
README or docs update for training and inference
```

The project should not move to every skill until clear satisfies the feasibility gate.

## 2026-07-16 Execution Status

The original distribution-calibrated grades did not pass qualitative review and
remain diagnostic only. A partial expert-guided v2 corrector also failed the
stronger requirement that its output be almost the same as an expert skeleton.

The current v3 implementation uses a full bone-preserving training-expert target
and a reference-conditioned Transformer. Checkpoint acceptance now requires all
validation students to improve, all corrected validation students to fall inside
the validation experts' leave-one-out Euclidean range, model-to-reference
distance below `0.10`, stable correction acceleration, and preserved bone
lengths. After re-extracting the automatically detected left-handed `EG28` and
`EG29` clips and retraining, the accepted epoch-54 checkpoint passed all ten
untouched test students:

```text
input nearest untouched-expert distance:     0.4209
corrected nearest untouched-expert distance: 0.1911
untouched expert leave-one-out mean:          0.2456
students inside untouched expert range:       10 / 10
mean model-to-training-reference distance:    0.0374
```

Handedness is now inferred before window selection by comparing scale-normalized
robust peak wrist speed and positive acceleration. A side is accepted only at a
motion-score ratio of at least `2.0`; ambiguous clips use the explicit override
file and then filename metadata. `EG28` and `EG29` were identified as left-handed
at ratios `6.41` and `6.58`, respectively.

Inference now also emits `keypoint_scores.csv` and `advice_context.jsonl`.
These outputs relate the diagnostic total to dominant/non-dominant keypoints,
distance components, motion phases, and correction directions so a language
model can produce evidence-grounded advice. The keypoint scores are explanatory
signals under the same calibration, not additive rubric points.

Current artifacts:

```text
models/skeleton_correction/clear_expert_guided_v3.pt
stats/skeleton_correction/clear_expert_guided_v3_training/
stats/skeleton_correction/clear_debug_videos/EG12_expert_guided_v3_test_h264.mp4
stats/skeleton_correction/clear_expert_guided_v3_grades/keypoint_scores.csv
stats/skeleton_correction/clear_expert_guided_v3_grades/advice_context.jsonl
```

This completes the expert-like skeleton geometry milestone for clear. It does
not complete score calibration: skill expansion and production grading remain
blocked on held-out per-video human scores.

## Agent-Mode Prompt

Use this prompt for a new agent or agentic team:

```text
You are working in /Users/heavenchen/dev/badminton-analysis.

Read docs/skeleton-correction-execution-plan.md and execute it end to end.

Start with badminton clear only. You may use the user's nislab server for heavy pose extraction, model training, and scoring runs. Sync code to nislab with rsync, run remote commands from ~/heaven/badminton-analysis, and copy generated stats/model artifacts back into this repo.

Do not implement every skill immediately. First prove feasibility for clear:
1. establish current baseline clear beginner/expert score distributions,
2. extract normalized skeleton-sequence datasets,
3. train a skeleton denoising/correction model on expert clear skeletons,
4. score beginner and expert clear videos by correction distance,
5. generate overlays showing original vs corrected skeletons,
6. compare old vs new scores,
7. calibrate so clear experts score around 98-100 and clear beginners/students score around 40-50,
8. only then decide whether to integrate the scorer into the CLI and extend to other skills.

Keep the current grading CLI behavior compatible. Do not remove the existing rule/prototype scorer. Add tests for normalization, scoring monotonicity, and CLI compatibility. Report feasibility clearly before expanding beyond clear.
```

## Operating Notes For The Agent

- Prefer small, inspectable commits or work chunks.
- Keep remote artifacts under `stats/skeleton_correction/`, `datasets/skeleton_sequences/`, and `models/skeleton_correction/`.
- Avoid syncing large generated frontend directories.
- Preserve user changes in the working tree.
- If full `stats/` synchronization is needed, use `rsync -avP` so interrupted transfers can resume.
- Document any videos that fail pose extraction instead of silently dropping them.
