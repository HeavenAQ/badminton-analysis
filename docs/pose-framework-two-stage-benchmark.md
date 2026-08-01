# Two-Stage Pose and Grading Benchmark

## Purpose

This benchmark selects the pose pipeline used by the badminton grading service.
It does not select a model from generic Human3.6M or COCO leaderboard numbers.
The decision is based on the repository's own serve, lift, clear, and smash
videos and the grading behavior those poses produce.

The benchmark has two independent stages:

1. Pose and checkpoint extraction is compared with a high-capacity 2D
   pseudo-labeler.
2. The extracted motion is passed through one fixed, expert-referenced grading
   protocol and compared on held-out score behavior.

Pseudo labels are not human ground truth. They make the comparison repeatable,
but qualitative overlays and cohort labels remain required checks.

## Frozen Corpus and Splits

The generated `manifest.csv` contains 81 clips selected deterministically from
the 400 successfully extracted local clips:

| Skill | Beginners | Experts | Total |
| --- | ---: | ---: | ---: |
| Serve | 8 | 12 | 20 |
| Lift | 8 | 12 | 20 |
| Clear | 9 | 12 | 21 |
| Smash | 8 | 12 | 20 |

Clear explicitly includes the known left-handed students EG28 and EG29. These
are the only left-handed labels in the frozen corpus; all other clips are
right-handed. For
each skill, eight experts are marked as the training-bank subset and four are
interleaved evaluation clips. The scoring benchmark uses all 12 known experts
as the available reference catalog and scores each expert leave-one-out, so a
clip can never select itself. The first four naturally sorted beginner clips
are used only to fit the monotonic score calibration; the remaining beginners
form the held-out score test set. No beginner test score is used to fit a
calibration.

## Stage 1: Pseudo-Label and Checkpoint Agreement

### Reference choice

[Sapiens2](https://arxiv.org/abs/2604.21681) is the preferred reference because
its pose head predicts 308 whole-body points at native 1K resolution. Its
[repository](https://github.com/facebookresearch/sapiens2),
[pose inference guide](https://github.com/facebookresearch/sapiens2/blob/main/docs/POSE.md),
and [license](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md)
are public, but its task checkpoints require accepted, authenticated access on
Hugging Face. The Nislab account had neither `HF_TOKEN` nor a cached token, and
the model API returned HTTP 401. The benchmark must not call an unavailable
checkpoint successfully evaluated.

The approved fallback is `YOLO26x-pose` from
[Ultralytics YOLO26](https://arxiv.org/abs/2606.03748), using the canonical
[Ultralytics repository](https://github.com/ultralytics/ultralytics) and
[pose documentation](https://docs.ultralytics.com/tasks/pose/). It returns the
same 17 COCO body joints used by the grading pipeline. Reference inference uses
the PyTorch checkpoint at 1280 pixels with a maximum of four people per frame;
the largest detected person is retained as the foreground athlete.

### Candidate-owned decisions

Each candidate must retain its own 2D coordinates and confidences, infer its
own handedness, and find its own analysis window. Candidate files may not copy
the production RTMW3D frame indices or 2D skeleton. Doing so would leak the
baseline decision into the model being measured.

Known corpus handedness metadata is the Stage 1 reference. This prevents a
single-model left/right swap from becoming false ground truth; for example,
YOLO26x inferred serve expert 19 as left-handed, and the prior RTMW extractor
inferred smash student EG31 as left-handed, even though video inspection shows
the racket in the anatomical right hand in both clips. Candidates must still infer
handedness from their own tracks. The same production window code then runs on
the reference and candidate coordinates:

1. Scale-normalized left and right wrist motion is smoothed.
2. Peak positive acceleration and peak speed determine the likely racket hand.
3. The skill-specific wrist and elbow rule selects start, peak, and end.
4. Five checkpoint anchors are start, start-to-peak midpoint, peak,
   peak-to-end midpoint, and end.

### Metrics

The stage reports every metric separately:

- handedness agreement;
- five-checkpoint mean absolute error in frames and milliseconds;
- checkpoint recall within 100 milliseconds;
- analysis-window intersection over union;
- phase-aligned normalized 2D joint distance;
- PCK at 0.2 normalized segment lengths;
- shared-joint angle mean absolute error in degrees;
- extraction failures.

No weighted aggregate hides a regression. Timing and handedness are gating
metrics because an accurate pose on the wrong frames or wrong body side cannot
produce valid criterion scores.

## Stage 2: Held-Out Grading Behavior

The score oracle and every candidate use the same transparent shadow scorer:

1. Phase-align the sequence to the five canonical checkpoints.
2. Adapt every eligible expert pose to the subject's observed bone lengths. A
   reference expert is scored leave-one-out.
3. Select the adapted expert with the lowest confidence-masked, skill-weighted
   position, angle, velocity, and bone-length correction distance.
4. Retain that same correction distance as the grading evidence.
5. Fit the existing monotonic exponential calibration on leave-one-out expert
   distances and four calibration beginners only, targeting expert mean 99 and
   beginner mean 45.
6. Calculate the skill's criterion scores from their configured frame windows
   and measured joint sets.

The shadow scorer isolates pose-model effects and is used for model selection.
After selection, the winner is retrained with the production correction model
and audited again before deployment.

Held-out metrics are:

- total-score MAE against the YOLO26x pseudo-label score oracle;
- criterion-score MAE;
- Spearman score rank correlation;
- expert-versus-beginner ROC AUC;
- leave-one-out known-expert and held-out beginner score means;
- calibration reachability and extraction failures.

Candidate-specific calibration is required because 2D and 3D coordinate
distributions have different raw distance scales. The split is identical for
all candidates, and only held-out results decide the winner.

## Candidate References

| Candidate | Paper | Repository | Benchmark role |
| --- | --- | --- | --- |
| RTMW3D-X | [RTMW](https://arxiv.org/abs/2407.08634) | [MMPose RTMPose3D](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d) | Current direct 3D baseline |
| MotionBERT | [MotionBERT](https://arxiv.org/abs/2210.06551) | [Official code](https://github.com/Walter0807/MotionBERT) | RTMPose-M 2D plus temporal 3D lifting |
| VideoPose3D-243 | [VideoPose3D](https://arxiv.org/abs/1811.11742) | [Official code](https://github.com/facebookresearch/VideoPose3D) | RTMPose-M 2D plus 243-frame temporal convolutional lifting |
| MediaPipe Heavy | [BlazePose](https://arxiv.org/abs/2006.10204) | [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Direct lightweight world-coordinate baseline |
| PoseMamba-L | [PoseMamba](https://arxiv.org/abs/2408.03540) | [Official code](https://github.com/nankingjing/PoseMamba) | Reproducibility screen |
| MotionAGFormer | [MotionAGFormer](https://arxiv.org/abs/2310.16288) | [Official code](https://github.com/TaatiTeam/MotionAGFormer) | Documented temporal alternative |
| PersPose | [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Hao_PersPose_3D_Human_Pose_Estimation_with_Perspective_Encoding_and_Perspective_ICCV_2025_paper.html) | [Official code](https://github.com/KenAdamsJoseph/PersPose) | Screened for deployment fit |

PoseMamba-L requires its custom `selective_scan_cuda_oflex` extension. Its
official environment targets Python 3.8, PyTorch 1.13, CUDA 11.7, while the
Nislab runtime has no full CUDA compiler (`nvcc`). It is recorded as not
reproducible in this deployment environment rather than assigned fabricated
performance numbers. PersPose requires SMPL assets and focal-length inputs and
does not provide the required real-time video path, so it is not ranked as a
drop-in service candidate.

[AthletePose3D](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/html/Yeung_AthletePose3D_A_Benchmark_Dataset_for_3D_Human_Pose_Estimation_and_CVPRW_2025_paper.html)
is an important interpretation constraint: generic monocular models can fail
on high-acceleration athletic motion even when their standard benchmark result
is strong. This is why local badminton checkpoint and score behavior is the
selection target.

## Reproduction

The executable harness is `scripts/benchmark_pose_frameworks.py`. Generated
NPZ, CSV, logs, and MP4 files stay outside Git under the dated local benchmark
directory. The final report records the server, GPU, package versions, model
hashes, exact commands, raw CSV paths, qualitative video paths, and selected
candidate.

The recorded Nislab environment is:

| Component | Value |
| --- | --- |
| Host | `p920`, Linux 6.8.0-124-generic x86_64 |
| Benchmark GPU | Quadro RTX 6000, 24 GiB (CUDA device 1; `nvidia-smi` index 0) |
| Driver | 595.71.05 |
| Python | 3.12.13 |
| PyTorch | 2.5.1+cu124, cuDNN 9.1 |
| MMPose | 1.3.2 |
| Ultralytics | 8.4.75 |
| OpenCV | 4.10.0 |
| YOLO26x-pose SHA-256 | `08ed9e01d22a6f248b04f2f9992016aca9a32250b9ab57057d886a09d026700d` |
| MediaPipe Heavy task SHA-256 | `64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b` |

Checked-out research commits are MotionBERT `705d3a9`, MotionAGFormer
`4756fd1`, PoseMamba `df38d59`, PersPose `43eb2c2`, and Sapiens2 `7e5bae8`.
The complete hashes remain in the raw environment record.

## Results

### Stage 1

All 81 clips completed for every ranked candidate with zero extraction
failures. These are corpus-wide means against the YOLO26x-pose pseudo labels:

| Candidate | Checkpoint MAE (ms) | Recall at 100 ms | Window IoU | Handedness | 2D distance | PCK@0.2 | Angle MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTMW3D-X | 80.74 | 0.8420 | 0.8973 | 0.9877 | 0.1502 | 0.8068 | 6.38 deg |
| MotionBERT | 64.69 | 0.8815 | 0.9196 | 0.9630 | 0.1277 | 0.8660 | 5.63 deg |
| **VideoPose3D-243** | **58.68** | **0.8938** | **0.9245** | 0.9753 | **0.1216** | **0.8729** | **5.23 deg** |
| MediaPipe Heavy | 113.00 | 0.7975 | 0.8716 | 0.9506 | 0.1919 | 0.7034 | 8.10 deg |

VideoPose3D is the Stage 1 leader. RTMW3D has the best handedness agreement,
including both known left-handed clear students, but its temporal checkpoint
agreement is weaker.

### Stage 2

The held-out set contains 48 leave-one-out experts and 17 beginners. Scores
below are candidate-specific calibrated outputs compared with the frozen
YOLO26x score oracle:

| Candidate | Total MAE | Criterion MAE | Spearman rho | Expert/beginner AUC | Expert mean | Beginner mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26x oracle | 0.00 | 0.00 | 1.0000 | 0.9167 | 97.17 | 44.83 |
| **RTMW3D-X** | **9.77** | **2.11** | **0.7811** | **0.9498** | **94.99** | 30.13 |
| MotionBERT | 29.13 | 5.51 | 0.2384 | 0.7181 | 72.95 | 42.89 |
| VideoPose3D-243 | 24.10 | 4.54 | 0.3709 | 0.7770 | 77.66 | 38.97 |
| MediaPipe Heavy | 14.74 | 2.91 | 0.6395 | 0.8983 | 89.13 | 25.31 |

These means belong to the benchmark's intentionally simple shadow scorer.
They are not production grades and are not expected to satisfy the production
99.8/45 cohort calibration.

### Runtime

The direct RTMW3D extractor processed the full 100-clip skill datasets at
72.95 FPS for serve, 57.50 FPS for lift, 50.13 FPS for clear, and 51.48 FPS for
smash. The temporal candidate benchmark, including shared RTMPose-M 2D
extraction and lifting, measured:

| Candidate | Mean seconds/clip | P95 seconds/clip | Aggregate FPS |
| --- | ---: | ---: | ---: |
| MotionBERT | 15.30 | 19.15 | 9.38 |
| VideoPose3D-243 | 14.65 | 18.26 | 9.80 |
| MediaPipe Heavy | **9.31** | **11.87** | **15.43** |

Runtime values are offline end-to-end extraction timings, not isolated neural
network kernel timings. Production TensorRT service latency is measured
separately after deployment. Nislab exposes the A6000 as CUDA device 0 even
though `nvidia-smi` labels it index 1; the benchmark commands selected CUDA
device 1 and therefore ran on the Quadro RTX 6000.

## Selection

RTMW3D-X remains the production pose framework. VideoPose3D wins the first
stage, so it is a credible frame-extraction alternative, but it loses the
decision metric: downstream grading. RTMW3D has 60% lower total-score MAE and
53% lower criterion-score MAE than VideoPose3D, substantially higher rank
correlation and cohort AUC, better handedness agreement, and materially higher
throughput in the existing direct 3D path. Selecting from Stage 1 alone would
optimize pseudo-label imitation while regressing the actual product behavior.

The production correctors were retrained only after this selection. Their full
100-clip-per-skill audits produced:

| Skill | Student mean | Expert mean | AUC | Test correction/expert distance | Test acceleration ratio | Improved students |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Serve | 45.000 | 99.801 | 0.99 | 0.275 | 0.997 | 100% |
| Lift | 45.000 | 99.803 | 1.00 | 0.356 | 0.995 | 100% |
| Clear | 45.000 | 99.795 | 1.00 | 0.312 | 0.988 | 100% |
| Smash | 45.000 | 99.805 | 0.98 | 0.319 | 0.995 | 100% |

The distance column compares the corrected student against its nearest
bone-length-adapted expert. The acceleration ratio compares corrected motion
with that adapted target; a value near one confirms that correction did not
obtain a low distance by collapsing temporal motion.

## Artifacts

Raw local outputs are intentionally ignored by Git:

- `stats/pose-framework-benchmark-20260801/manifest.csv`
- `stats/pose-framework-benchmark-20260801/stage1-final-details.csv`
- `stats/pose-framework-benchmark-20260801/stage1-final-summary.csv`
- `stats/pose-framework-benchmark-20260801/stage2-final-details.csv`
- `stats/pose-framework-benchmark-20260801/stage2-final-summary.csv`
- `stats/pose-framework-benchmark-20260801/qualitative/*.mp4`
- `stats/production-correction-20260801/grades/*/all_grades.csv`
- `stats/production-correction-20260801/training/*/expert_distance_summary.json`

Each qualitative folder contains one held-out beginner and one expert video per
skill. The source frame appears beside RTMW3D, MotionBERT, VideoPose3D, and
MediaPipe normalized 3D tracks.

Portable model SHA-256 values are:

| Model | SHA-256 |
| --- | --- |
| RTMW3D-X ONNX | `4a289c0e99d47eb595e99679d9d4a2d1def1b4241f9adcbafba44b9ff585ebcd` |
| YOLOX-M detector ONNX | `3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23` |
| MotionBERT checkpoint | `d80af32396c60cf66fa5afb7ef7f7c869ae0851afd3d91a75d55e76c5a62cb23` |
| VideoPose3D checkpoint | `88f5abbb4e37499781d5646665f7c46b521d139b7fb54d182913c19a76b9c6de` |
| Serve corrector ONNX | `5243e7c7f7abeea94985cb6aec9034a9fb4e28832ad5ff21d9bcee59d987ea28` |
| Lift corrector ONNX | `1843bae6a6232ca58efc4f07e30de09c50527bc520c9ebb36b54039c1987fd30` |
| Clear corrector ONNX | `640be0ed09eb5b6d7de1c4c3fa52a15361539746f146d0d43b987f86feed7019` |
| Smash corrector ONNX | `89b4abb03461705aaf1c081c9f0a7d71d5e52e346db0e7c128418839fe9cc111` |

## Reproduction Commands

The benchmark subcommands are deliberately separate so GPU-heavy extraction
cannot silently overlap evaluation:

```bash
python scripts/benchmark_pose_frameworks.py manifest \
  --source-root . --dataset-root datasets/skeleton_sequences \
  --output "$BENCHMARK/manifest.csv"

python scripts/benchmark_pose_frameworks.py extract-yolo-reference \
  --manifest "$BENCHMARK/manifest.csv" --output-root "$BENCHMARK" \
  --model yolo26x-pose.pt --device 0 --image-size 1280

python scripts/benchmark_pose_frameworks.py evaluate-frame-selection \
  --manifest "$BENCHMARK/manifest.csv" --output-root "$BENCHMARK" \
  --candidates rtmw3d motionbert videopose3d mediapipe \
  --details "$BENCHMARK/stage1-final-details.csv" \
  --summary "$BENCHMARK/stage1-final-summary.csv"

python scripts/benchmark_pose_frameworks.py evaluate-grading \
  --manifest "$BENCHMARK/manifest.csv" --output-root "$BENCHMARK" \
  --candidates rtmw3d motionbert videopose3d mediapipe \
  --details "$BENCHMARK/stage2-final-details.csv" \
  --summary "$BENCHMARK/stage2-final-summary.csv"
```

Candidate extraction commands require their official repositories and
checkpoints described above. The raw Nislab environment record contains those
checkout paths and complete Git revisions.
