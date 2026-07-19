from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from badminton_analysis.ml.models.skeleton_denoiser import SkeletonDenoiser
from badminton_analysis.ml.skeleton_dataset import discover_sequence_files, load_sequence
from badminton_analysis.ml.skeleton_normalization import (
    phase_align_sequence,
    restore_phase_timing,
)
from badminton_analysis.ml.skeleton_scoring import (
    ScoreCalibration,
    correction_distance,
    correction_quality_metrics,
    expert_euclidean_distances,
    fit_score_calibration,
    JOINT_WEIGHTS,
    keypoint_correction_components,
    project_bone_lengths,
)

DETAILS = (
    ("Preparation correction", 10.0, 0, 16, None),
    ("Rotation correction", 10.0, 8, 32, (5, 6, 11, 12, 13, 14, 15, 16)),
    ("Balance correction", 20.0, 16, 40, (5, 7, 9, 6, 8, 10)),
    ("Contact correction", 20.0, 27, 38, (6, 8, 10)),
    ("Wrist/arm correction", 20.0, 24, 48, (6, 8, 10)),
    ("Follow-through correction", 20.0, 40, 64, None),
)

ADVICE_KEYPOINTS = {
    0: "head",
    5: "non_dominant_shoulder",
    6: "dominant_shoulder",
    7: "non_dominant_elbow",
    8: "dominant_elbow",
    9: "non_dominant_wrist",
    10: "dominant_wrist",
    11: "non_dominant_hip",
    12: "dominant_hip",
    13: "non_dominant_knee",
    14: "dominant_knee",
    15: "non_dominant_ankle",
    16: "dominant_ankle",
}

KEYPOINT_PHASES = (
    ("preparation", 0, 16),
    ("rotation", 8, 32),
    ("contact", 27, 38),
    ("follow_through", 38, 64),
)


def load_corrector(
    path: str | Path, device: torch.device
) -> tuple[SkeletonDenoiser, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = SkeletonDenoiser(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    if checkpoint.get("reference_conditioned", False):
        model.set_expert_reference_bank(
            torch.as_tensor(
                checkpoint["expert_reference_skeletons"],
                dtype=torch.float32,
                device=device,
            ),
            torch.as_tensor(
                checkpoint["expert_reference_confidence"],
                dtype=torch.float32,
                device=device,
            ),
        )
    return model, checkpoint


def predict_correction(
    model: SkeletonDenoiser,
    skeleton: np.ndarray,
    confidence: np.ndarray,
    device: torch.device,
    phase_indices: np.ndarray | None = None,
    correction_strength: float = 1.0,
) -> np.ndarray:
    if not 0.0 <= correction_strength <= 1.0:
        raise ValueError("correction strength must be between zero and one")
    model_skeleton = skeleton
    model_confidence = confidence
    if phase_indices is not None:
        model_skeleton = phase_align_sequence(skeleton, phase_indices)
        model_confidence = np.clip(
            phase_align_sequence(confidence, phase_indices), 0.0, 1.0
        )
    feature_parts = [model_skeleton]
    if model.input_features == 7:
        reference_skeletons = model.expert_reference_skeletons
        reference_confidence = model.expert_reference_confidence
        if reference_skeletons is None or reference_confidence is None:
            raise ValueError("reference-conditioned model has no expert bank")
        expert_skeletons = reference_skeletons.detach().cpu().numpy()
        expert_confidence = reference_confidence.detach().cpu().numpy()
        distances = expert_euclidean_distances(
            model_skeleton,
            expert_skeletons,
            model_confidence,
            expert_confidence,
        )
        nearest_expert = expert_skeletons[int(np.argmin(distances))]
        feature_parts.append(
            project_bone_lengths(model_skeleton, nearest_expert)
        )
    feature_parts.append(model_confidence[..., None])
    features = np.concatenate(feature_parts, axis=-1)
    tensor = torch.as_tensor(features[None], dtype=torch.float32, device=device)
    with torch.inference_mode():
        output = model(tensor)[0].cpu().numpy()
    output = model_skeleton + correction_strength * (output - model_skeleton)
    corrected = project_bone_lengths(
        model_skeleton, np.asarray(output, dtype=np.float32)
    )
    if phase_indices is not None:
        correction = restore_phase_timing(
            corrected - model_skeleton, phase_indices
        )
        corrected = project_bone_lengths(skeleton, skeleton + correction)
    return corrected


def phase_grading_details(
    original: np.ndarray,
    corrected: np.ndarray,
    confidence: np.ndarray,
    calibration: ScoreCalibration,
    total_grade: float | None = None,
) -> list[tuple[str, float, float]]:
    output: list[tuple[str, float, float]] = []
    maxima: list[float] = []
    for description, maximum, start, end, joints in DETAILS:
        source = original[start:end]
        target = corrected[start:end]
        mask = confidence[start:end]
        if joints is not None:
            source = source[:, joints]
            target = target[:, joints]
            mask = mask[:, joints]
            # correction_distance expects the canonical joint weight vector.
            magnitude = np.linalg.norm(source - target, axis=-1)
            distance = float(np.sum(magnitude * mask) / max(1e-8, np.sum(mask)))
        else:
            distance, _ = correction_distance(source, target, mask)
        grade = maximum * float(calibration.score(distance)) / 100.0
        output.append((description, distance, grade))
        maxima.append(maximum)
    if total_grade is not None and output:
        raw_total = sum(item[2] for item in output)
        grades = [item[2] for item in output]
        if raw_total > total_grade and raw_total > 1e-8:
            grades = [grade * total_grade / raw_total for grade in grades]
        elif raw_total < total_grade:
            capacities = [maximum - grade for maximum, grade in zip(maxima, grades)]
            capacity_total = sum(capacities)
            if capacity_total > 1e-8:
                grades = [
                    grade + (total_grade - raw_total) * capacity / capacity_total
                    for grade, capacity in zip(grades, capacities)
                ]
        output = [
            (description, distance, grade)
            for (description, distance, _), grade in zip(output, grades)
        ]
    return output


def _correction_direction(vector: np.ndarray) -> str:
    if float(np.linalg.norm(vector)) < 0.01:
        return "minimal_change"
    axis = int(np.argmax(np.abs(vector)))
    if axis == 0:
        return "toward_dominant_side" if vector[0] > 0 else "toward_non_dominant_side"
    if axis == 1:
        return "raise" if vector[1] > 0 else "lower"
    return "forward" if vector[2] > 0 else "backward"


def keypoint_advice_details(
    original: np.ndarray,
    corrected: np.ndarray,
    confidence: np.ndarray,
    calibration: ScoreCalibration,
) -> list[dict[str, Any]]:
    full_metrics = keypoint_correction_components(
        original, corrected, confidence
    )
    phase_metrics = {
        name: keypoint_correction_components(
            original[start:end], corrected[start:end], confidence[start:end]
        )
        for name, start, end in KEYPOINT_PHASES
    }
    output: list[dict[str, Any]] = []
    for joint_index, keypoint_name in ADVICE_KEYPOINTS.items():
        phase_distances = {
            name: float(metrics["correction_distance"][joint_index])
            for name, metrics in phase_metrics.items()
        }
        worst_phase = max(phase_distances, key=phase_distances.__getitem__)
        phase_start, phase_end = next(
            (start, end)
            for name, start, end in KEYPOINT_PHASES
            if name == worst_phase
        )
        mask = confidence[phase_start:phase_end, joint_index].astype(np.float64)
        correction = (
            corrected[phase_start:phase_end, joint_index]
            - original[phase_start:phase_end, joint_index]
        ).astype(np.float64)
        denominator = float(np.sum(mask))
        vector = (
            np.sum(correction * mask[:, None], axis=0) / denominator
            if denominator > 1e-8
            else np.zeros(3, dtype=np.float64)
        )
        distance = float(
            full_metrics["correction_distance"][joint_index]
        )
        output.append(
            {
                "joint_index": joint_index,
                "keypoint": keypoint_name,
                "score": float(calibration.score(distance)),
                "correction_distance": distance,
                "position_distance": float(
                    full_metrics["position_distance"][joint_index]
                ),
                "angle_distance": float(
                    full_metrics["angle_distance"][joint_index]
                ),
                "velocity_distance": float(
                    full_metrics["velocity_distance"][joint_index]
                ),
                "bone_length_distance": float(
                    full_metrics["bone_length_distance"][joint_index]
                ),
                "importance_weight": float(JOINT_WEIGHTS[joint_index]),
                "worst_phase": worst_phase,
                "correction_direction": _correction_direction(vector),
                "correction_vector": [float(value) for value in vector],
                "phase_scores": {
                    name: float(calibration.score(value))
                    for name, value in phase_distances.items()
                },
                "phase_distances": phase_distances,
            }
        )
    return output


def _summary_rows(results: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expert_rows = results[results["label"] == "experts"]
    validation_experts = expert_rows[
        expert_rows["evaluation_split"] == "validation"
    ]
    separation_experts = validation_experts if len(validation_experts) else expert_rows
    beginner_rows = results[results["label"] == "beginners"]
    expert_min = float(separation_experts["total_grade"].min())
    expert_distances = separation_experts["correction_distance"].to_numpy(
        dtype=np.float64
    )
    beginner_distances = beginner_rows["correction_distance"].to_numpy(dtype=np.float64)
    pairwise = beginner_distances[:, None] - expert_distances[None, :]
    separation_auc = float(np.mean(pairwise > 0) + 0.5 * np.mean(pairwise == 0))
    expert_max_distance = float(np.max(expert_distances))
    for label, group in results.groupby("label", sort=False):
        scores = group["total_grade"].astype(float)
        distances = group["correction_distance"].astype(float)
        rows.append(
            {
                "label": label,
                "count": len(group),
                "mean": scores.mean(),
                "median": scores.median(),
                "min": scores.min(),
                "max": scores.max(),
                "overlap": (
                    float(np.mean(scores >= expert_min)) if label == "beginners" else 0.0
                ),
                "correction_distance_mean": distances.mean(),
                "correction_distance_median": distances.median(),
                "distance_overlap": (
                    float(np.mean(distances <= expert_max_distance))
                    if label == "beginners"
                    else 0.0
                ),
                "separation_auc": separation_auc,
            }
        )
    if len(validation_experts):
        scores = validation_experts["total_grade"].astype(float)
        distances = validation_experts["correction_distance"].astype(float)
        rows.append(
            {
                "label": "experts_validation",
                "count": len(validation_experts),
                "mean": scores.mean(),
                "median": scores.median(),
                "min": scores.min(),
                "max": scores.max(),
                "overlap": 0.0,
                "correction_distance_mean": distances.mean(),
                "correction_distance_median": distances.median(),
                "distance_overlap": 0.0,
                "separation_auc": separation_auc,
            }
        )
    return rows


def _load_old_scores(paths: list[str | None]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label, path in zip(("beginners", "experts"), paths):
        if not path or not Path(path).exists():
            continue
        frame = pd.read_csv(path)
        frame["label"] = label
        frames.append(frame[["filename", "label", "total_grade"]])
    if not frames:
        return pd.DataFrame(columns=["filename", "label", "total_grade"])
    return pd.concat(frames, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate skeleton correction feasibility")
    parser.add_argument("--dataset-root", default="datasets/skeleton_sequences/clear")
    parser.add_argument(
        "--model-path",
        default="models/skeleton_correction/clear_expert_guided_v3.pt",
    )
    parser.add_argument("--output-dir", default="stats/skeleton_correction/clear_feasibility")
    parser.add_argument("--baseline-beginners")
    parser.add_argument("--baseline-experts")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    model, checkpoint = load_corrector(args.model_path, device)
    validation_files = set(
        checkpoint.get(
            "expert_validation_files", checkpoint.get("validation_files", [])
        )
    )
    phase_aligned = bool(checkpoint.get("phase_aligned", False))
    correction_strength = float(checkpoint.get("inference_strength", 1.0))
    raw_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label in ("beginners", "experts"):
        for path in discover_sequence_files(args.dataset_root, label):
            sample = load_sequence(path)
            skeleton = sample["skeleton_3d"].astype(np.float32)
            confidence = sample["confidence"].astype(np.float32)
            corrected = predict_correction(
                model,
                skeleton,
                confidence,
                device,
                sample["phase_indices"] if phase_aligned else None,
                correction_strength,
            )
            distance, components = correction_distance(skeleton, corrected, confidence)
            quality = correction_quality_metrics(skeleton, corrected, confidence)
            filename = str(sample["video_name"].item())
            raw_rows.append(
                {
                    "filename": filename,
                    "label": label,
                    "handedness": str(sample["handedness"].item()),
                    "correction_distance": distance,
                    **components,
                    **quality,
                    "dataset_path": str(path),
                    "evaluation_split": (
                        "validation"
                        if label == "experts" and path.name in validation_files
                        else "training"
                        if label == "experts"
                        else "evaluation"
                    ),
                }
            )
            predictions[(label, filename)] = (skeleton, corrected, confidence)
    distances = pd.DataFrame(raw_rows)
    calibration_experts = distances[
        (distances["label"] == "experts")
        & (distances["evaluation_split"] == "validation")
    ]
    if calibration_experts.empty:
        calibration_experts = distances[distances["label"] == "experts"]
    calibration = fit_score_calibration(
        calibration_experts["correction_distance"].to_numpy(),
        distances.loc[distances["label"] == "beginners", "correction_distance"].to_numpy(),
    )

    grading_rows: list[dict[str, Any]] = []
    keypoint_rows: list[dict[str, Any]] = []
    advice_rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        skeleton, corrected, confidence = predictions[(raw["label"], raw["filename"])]
        total_grade = float(calibration.score(raw["correction_distance"]))
        sample = load_sequence(raw["dataset_path"])
        window = sample["analysis_window"].astype(int)
        row: dict[str, Any] = {
            "filename": raw["filename"],
            "skill": "clear",
            "handedness": raw["handedness"],
            "status": "success",
            "error": "",
            "total_grade": total_grade,
            "start_frame": int(window[0]),
            "peak_frame": int(window[1]),
            "end_frame": int(window[2]),
            "label": raw["label"],
            "evaluation_split": raw["evaluation_split"],
        }
        for index, (description, detail_distance, grade) in enumerate(
            phase_grading_details(
                skeleton,
                corrected,
                confidence,
                calibration,
                total_grade,
            ),
            start=1,
        ):
            row[f"detail_{index}_desc"] = description
            row[f"detail_{index}_grade"] = grade
            row[f"detail_{index}_distance"] = detail_distance
        keypoints = keypoint_advice_details(
            skeleton, corrected, confidence, calibration
        )
        for keypoint in keypoints:
            prefix = f"keypoint_{keypoint['keypoint']}"
            row[f"{prefix}_score"] = keypoint["score"]
            row[f"{prefix}_distance"] = keypoint["correction_distance"]
            vector = keypoint["correction_vector"]
            phase_scores = keypoint["phase_scores"]
            phase_distances = keypoint["phase_distances"]
            keypoint_rows.append(
                {
                    "filename": raw["filename"],
                    "subject_name": Path(raw["filename"]).stem,
                    "label": raw["label"],
                    "handedness": raw["handedness"],
                    "total_grade": total_grade,
                    **{
                        key: value
                        for key, value in keypoint.items()
                        if key
                        not in (
                            "correction_vector",
                            "phase_scores",
                            "phase_distances",
                        )
                    },
                    "correction_vector_x": vector[0],
                    "correction_vector_y": vector[1],
                    "correction_vector_z": vector[2],
                    **{
                        f"{name}_score": score
                        for name, score in phase_scores.items()
                    },
                    **{
                        f"{name}_distance": distance
                        for name, distance in phase_distances.items()
                    },
                    "score_status": "diagnostic_group_calibrated",
                }
            )
        priorities = sorted(
            keypoints,
            key=lambda detail: (
                float(detail["score"]),
                -float(detail["importance_weight"]),
            ),
        )[:5]
        advice_rows.append(
            {
                "filename": raw["filename"],
                "subject_name": Path(raw["filename"]).stem,
                "group": raw["label"],
                "handedness": raw["handedness"],
                "total_grade": total_grade,
                "score_status": "diagnostic_group_calibrated",
                "score_relationship": (
                    "Total grade is the calibrated weighted sequence correction. "
                    "Keypoint scores use the same calibration on joint-attributed "
                    "position, velocity, angle, and bone correction distance; "
                    "they are evidence for advice, not arithmetic grade parts."
                ),
                "priority_corrections": priorities,
                "keypoints": keypoints,
            }
        )
        row.update(
            correction_distance=raw["correction_distance"],
            position_distance=raw["position_distance"],
            angle_distance=raw["angle_distance"],
            velocity_distance=raw["velocity_distance"],
            bone_length_distance=raw["bone_length_distance"],
            model_path=str(args.model_path),
            scorer="skeleton-correction",
            score_status="diagnostic_group_calibrated",
        )
        grading_rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grading = pd.DataFrame(grading_rows)
    grading.to_csv(output_dir / "grading_results.csv", index=False)
    grade_export = grading[
        [
            "filename",
            "label",
            "total_grade",
            "evaluation_split",
            "status",
            "model_path",
            "score_status",
        ]
    ].copy()
    grade_export.insert(
        0,
        "subject_name",
        grade_export["filename"].map(lambda name: Path(name).stem),
    )
    grade_export = grade_export.rename(
        columns={"label": "group", "total_grade": "grade"}
    )
    grade_export.to_csv(output_dir / "all_grades.csv", index=False)
    pd.DataFrame(keypoint_rows).to_csv(
        output_dir / "keypoint_scores.csv", index=False
    )
    (output_dir / "advice_context.jsonl").write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False) for row in advice_rows
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(_summary_rows(grading)).to_csv(output_dir / "score_summary.csv", index=False)
    distances.to_csv(output_dir / "distance_components.csv", index=False)
    old_scores = _load_old_scores([args.baseline_beginners, args.baseline_experts])
    comparison = grading[["filename", "label", "total_grade"]].rename(
        columns={"total_grade": "new_total_grade"}
    ).merge(
        old_scores.rename(columns={"total_grade": "old_total_grade"}),
        on=["filename", "label"],
        how="left",
    )
    comparison["score_delta"] = comparison["new_total_grade"] - comparison["old_total_grade"]
    comparison.to_csv(output_dir / "old_vs_new_scores.csv", index=False)
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration.to_dict(), indent=2), encoding="utf-8"
    )
    Path(args.model_path).with_suffix(".calibration.json").write_text(
        json.dumps(calibration.to_dict(), indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(_summary_rows(grading)).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
