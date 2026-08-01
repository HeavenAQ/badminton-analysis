# Qualitative Criteria And RTMW3D Retraining Verification

## Scope

This audit was run on 2026-08-02 after restoring the historical qualitative
coaching criteria and aligning training extraction with the production RTMW3D
pose provider. No degree or angle threshold is included in coaching output.

The canonical ordered criteria are defined in
`badminton_analysis/ml/skill_specs.py` and documented in
`docs/skeleton-correction-pipeline.md`.

## Data Boundary

Only the `left` and `right` directories under each NSTC skill were added. No
person-named NSTC directory was scanned. Serve excludes the legacy teammate
expert directory and uses NSTC experts only.

| Skill | Legacy right experts | NSTC right | NSTC left | Total experts | Extraction failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serve | 0 | 16 | 10 | 26 | 0 |
| Lift | 50 | 22 | 10 | 82 | 0 |
| Clear | 50 | 30 | 20 | 100 | 0 |
| Smash | 50 | 16 | 9 | 75 | 0 |

The production-provider extraction summaries are stored under
`stats/rtmw3d-qualitative-retraining-20260802/extraction/`.

## Checkpoints

Each skill was trained independently for 150 epochs. The accepted checkpoint
is the epoch with the best validation expert distance that also passed every
correction, smoothness, bone-length, and expert-range gate.

| Skill | Accepted epoch | Right references | Left references |
| --- | ---: | ---: | ---: |
| Serve | 48 | 11 | 7 |
| Lift | 71 | 51 | 7 |
| Clear | 51 | 56 | 14 |
| Smash | 70 | 46 | 6 |

Every checkpoint embeds its skill-specific qualitative rules and authoritative
reference handedness. Inference raises an error when the requested hand has no
matching reference; it never falls back to the opposite hand.

## Score Audit

Calibration targets a beginner cohort mean of 45.0 and expert cohort mean of
99.8. Results below use all extracted samples, including held-out validation
and test experts.

| Skill | Beginner mean | Left expert mean | Left expert minimum | Right expert mean |
| --- | ---: | ---: | ---: | ---: |
| Serve | 45.00 | 100.00 | 100.00 | 99.50 |
| Lift | 45.00 | 99.52 | 95.16 | 99.80 |
| Clear | 45.00 | 99.52 | 90.35 | 99.77 |
| Smash | 45.00 | 100.00 | 100.00 | 99.77 |

The complete per-video grades and handedness summary are stored in:

```text
stats/rtmw3d-qualitative-retraining-20260802/grading/all_skill_scores.csv
stats/rtmw3d-qualitative-retraining-20260802/grading/handedness_score_summary.csv
```

## Corrected-To-Expert Euclidean Audit

For every expert sample, the audit restricts selection to same-handed training
references, adapts the selected reference to the source bone lengths, and
calculates mean 3D joint Euclidean distance after phase alignment. All 283
expert samples selected a same-handed reference.

| Skill | Hand | Corrected mean | Corrected maximum |
| --- | --- | ---: | ---: |
| Serve | Left | 0.0128 | 0.0153 |
| Serve | Right | 0.0137 | 0.0205 |
| Lift | Left | 0.0089 | 0.0134 |
| Lift | Right | 0.0087 | 0.0198 |
| Clear | Left | 0.0178 | 0.0408 |
| Clear | Right | 0.0161 | 0.0273 |
| Smash | Left | 0.0105 | 0.0163 |
| Smash | Right | 0.0114 | 0.0212 |

These are normalized body-coordinate distances. The full per-video audit,
including selected reference IDs and original distances, is stored in:

```text
stats/rtmw3d-qualitative-retraining-20260802/grading/corrected_expert_euclidean_audit.csv
```

## Automated Verification

- Source Python tests: 115 passed.
- Production Python tests: 83 passed.
- Go service tests: all packages passed.
- All four ONNX exports passed PyTorch-versus-ONNX output verification.
- Production deployment additionally analyzes held-out left-handed serve expert
  `nstc_left_IMG_5568`, requires a grade of at least 99, requires an
  `nstc_left_` expert match, and verifies TensorRT activation plus signed video
  playback before traffic promotion.
